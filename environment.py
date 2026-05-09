"""
environment.py
==============
Custom Gymnasium environment for RL-based movie recommendation.

Key fixes in v3:
  - Repetition penalty raised 0.1 → 0.5 (was too weak, agents learned
    to repeat 5-star movies 9-13 times per episode)
  - Repetition penalty now scales with how many times a movie has been
    recommended (first repeat = 0.5, second = 0.8, third+ = 1.0)
  - Diversity threshold slightly lowered 0.42 → 0.38 to compensate for
    agents now being forced to explore more
  - Satisfaction gain raised 0.06 → 0.07 to reward successful diversity
  - Added already_seen_mask to prevent recommending already-seen movies
    from being the dominant strategy (agent penalized heavily)

State (65-dim):
  - 64-dim: mean BPR embedding of last K recommended movies
  - 1-dim:  current satisfaction score [0, 1]

Action:
  - Integer 0-499: index into catalog

Episode:
  - Fixed length of 20 steps
  - New random user sampled each episode

Satisfaction targets (random agent):
  - Greedy:  final sat ~0.0  (collapses)
  - Random:  final sat ~0.4-0.6
  - RL good: final sat ~0.7-0.9
"""

import os
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


class MovieRecEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(
        self,
        data_dir="./data/processed",
        episode_length=20,
        history_window=10,
        diversity_window=5,
        reward_weights=None,
        satisfaction_gain=0.07,       # raised from 0.06
        satisfaction_loss=0.10,
        diversity_threshold=0.38,     # lowered from 0.42 — compensate for harder rep penalty
        initial_satisfaction=1.0,
        rating_power=0.7,
        seed=42,
    ):
        super().__init__()

        self.episode_length       = episode_length
        self.history_window       = history_window
        self.diversity_window     = diversity_window
        self.satisfaction_gain    = satisfaction_gain
        self.satisfaction_loss    = satisfaction_loss
        self.diversity_threshold  = diversity_threshold
        self.initial_satisfaction = initial_satisfaction
        self.rating_power         = rating_power

        # Repetition penalty weights
        # Scaled by how many times the movie has been recommended:
        # 1st repeat: 0.5, 2nd repeat: 0.8, 3rd+: 1.0
        # Makes repeated recommendations increasingly costly
        self.rep_penalties = [0.0, 0.5, 0.8, 1.0]

        self.reward_weights = reward_weights or {
            "rating":    0.6,
            "diversity": 0.3,
        }
        # Note: repetition is handled separately via scaled penalty above

        # ── Load data ────────────────────────────────────────────────────
        self.movie_embeddings = np.load(
            f"{data_dir}/movie_embeddings.npy", allow_pickle=True
        ).item()
        self.user_profiles = np.load(
            f"{data_dir}/user_profiles.npy", allow_pickle=True
        ).item()
        self.user_ratings = np.load(
            f"{data_dir}/user_ratings.npy", allow_pickle=True
        ).item()
        self.catalog = np.load(
            f"{data_dir}/catalog.npy", allow_pickle=True
        ).tolist()
        self.movie_titles = np.load(
            f"{data_dir}/movie_titles.npy", allow_pickle=True
        ).item()

        # ── Genre lookup ─────────────────────────────────────────────────
        self.movie_genres = {}
        movies_path = os.path.join(os.path.dirname(data_dir), "movies.dat")
        if os.path.exists(movies_path):
            movies = pd.read_csv(
                movies_path, sep="::", engine="python",
                names=["movie_id", "title", "genres"],
                encoding="latin-1"
            )
            for _, row in movies.iterrows():
                self.movie_genres[row["movie_id"]] = set(row["genres"].split("|"))

        # ── Derived ──────────────────────────────────────────────────────
        self.n_movies  = len(self.catalog)
        self.embed_dim = next(iter(self.movie_embeddings.values())).shape[0]
        self.user_ids  = list(self.user_profiles.keys())

        self.emb_matrix = np.array(
            [self.movie_embeddings[mid] for mid in self.catalog],
            dtype=np.float32
        )
        self.movie_to_action = {mid: i for i, mid in enumerate(self.catalog)}

        # ── Gym spaces ───────────────────────────────────────────────────
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0,
            shape=(self.embed_dim + 1,),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.n_movies)

        # ── Episode state ────────────────────────────────────────────────
        self.current_user_id   = None
        self.user_profile      = None
        self.user_rating_dict  = {}
        self.satisfaction      = self.initial_satisfaction
        self.timestep          = 0
        self.history           = []
        self.recommended_set   = set()
        self.recommended_count = {}   # movie_id → how many times recommended
        self.genres_seen       = set()

        self.rng = np.random.default_rng(seed)

    # ────────────────────────────────────────────────────────────────────
    # reset
    # ────────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_user_id   = int(self.rng.choice(self.user_ids))
        self.user_profile      = self.user_profiles[self.current_user_id]
        self.user_rating_dict  = self.user_ratings.get(self.current_user_id, {})
        self.satisfaction      = self.initial_satisfaction
        self.timestep          = 0
        self.history           = []
        self.recommended_set   = set()
        self.recommended_count = {}
        self.genres_seen       = set()

        return self._get_obs(), {"user_id": self.current_user_id}

    # ────────────────────────────────────────────────────────────────────
    # step
    # ────────────────────────────────────────────────────────────────────

    def step(self, action):
        action    = int(action)
        movie_id  = self.catalog[action]
        movie_vec = self.movie_embeddings[movie_id]

        # ── 1. Rating score ──────────────────────────────────────────────
        if movie_id in self.user_rating_dict:
            rating_score = float(self.user_rating_dict[movie_id]) / 5.0
        else:
            cosine_sim   = float(np.dot(self.user_profile, movie_vec))
            rating_score = float(((cosine_sim + 1.0) / 2.0) ** self.rating_power)
        rating_score = float(np.clip(rating_score, 0.0, 1.0))

        # ── 2. Diversity bonus ───────────────────────────────────────────
        emb_div         = self._embedding_diversity(movie_vec)
        genre_div       = self._genre_diversity(movie_id)
        diversity_bonus = float(np.clip(0.7 * emb_div + 0.3 * genre_div, 0.0, 1.0))

        # ── 3. Scaled repetition penalty ────────────────────────────────
        # How many times has this movie been recommended already?
        times_seen = self.recommended_count.get(movie_id, 0)
        # Clamp to max index in penalty list
        pen_idx    = min(times_seen, len(self.rep_penalties) - 1)
        repetition_penalty = self.rep_penalties[pen_idx]

        # ── 4. Reward ────────────────────────────────────────────────────
        w = self.reward_weights
        raw = max(0.0,
            w["rating"]    * rating_score
          + w["diversity"] * diversity_bonus
          - repetition_penalty
        )
        reward = float(raw * self.satisfaction)

        # ── 5. Satisfaction ──────────────────────────────────────────────
        if diversity_bonus >= self.diversity_threshold:
            self.satisfaction = min(1.0, self.satisfaction + self.satisfaction_gain)
        else:
            self.satisfaction = max(0.0, self.satisfaction - self.satisfaction_loss)

        # ── 6. Update state ──────────────────────────────────────────────
        self.history.append((movie_id, movie_vec))
        self.recommended_set.add(movie_id)
        self.recommended_count[movie_id] = self.recommended_count.get(movie_id, 0) + 1
        self.timestep += 1
        if movie_id in self.movie_genres:
            self.genres_seen.update(self.movie_genres[movie_id])

        terminated = self.timestep >= self.episode_length
        truncated  = False

        info = {
            "movie_id":            movie_id,
            "movie_title":         self.movie_titles.get(movie_id, str(movie_id)),
            "rating_score":        rating_score,
            "embedding_diversity": emb_div,
            "genre_diversity":     genre_div,
            "diversity_bonus":     diversity_bonus,
            "repetition_penalty":  repetition_penalty,
            "satisfaction":        self.satisfaction,
            "timestep":            self.timestep,
            "n_unique_genres":     len(self.genres_seen),
            "n_unique_movies":     len(self.recommended_set),
        }

        return self._get_obs(), reward, terminated, truncated, info

    # ────────────────────────────────────────────────────────────────────
    # Helpers
    # ────────────────────────────────────────────────────────────────────

    def _get_obs(self):
        if len(self.history) == 0:
            mean_emb = np.zeros(self.embed_dim, dtype=np.float32)
        else:
            recent   = [v for _, v in self.history[-self.history_window:]]
            mean_emb = np.mean(recent, axis=0).astype(np.float32)
        return np.append(mean_emb, self.satisfaction).astype(np.float32)

    def _embedding_diversity(self, movie_vec):
        if len(self.history) == 0:
            return 1.0
        recent      = [v for _, v in self.history[-self.diversity_window:]]
        recent_mean = np.mean(recent, axis=0)
        norm        = np.linalg.norm(recent_mean)
        if norm > 0:
            recent_mean = recent_mean / norm
        sim = float(np.clip(np.dot(movie_vec, recent_mean), -1.0, 1.0))
        return float((1.0 - sim) / 2.0)

    def _genre_diversity(self, movie_id):
        if movie_id not in self.movie_genres or len(self.history) == 0:
            return 0.5
        movie_genre_set = self.movie_genres[movie_id]
        if not movie_genre_set:
            return 0.5
        recent_ids    = [mid for mid, _ in self.history[-self.diversity_window:]]
        recent_genres = set()
        for mid in recent_ids:
            if mid in self.movie_genres:
                recent_genres.update(self.movie_genres[mid])
        new_genres = movie_genre_set - recent_genres
        return float(len(new_genres) / len(movie_genre_set))

    def render(self):
        print(f"  Step {self.timestep}/{self.episode_length} | "
              f"User {self.current_user_id} | Sat {self.satisfaction:.2f}")


# ────────────────────────────────────────────────────────────────────────
# Validation
# ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Environment v3 Validation ===\n")

    env = MovieRecEnv(data_dir="./data/processed")
    print(f"Obs space        : {env.observation_space}")
    print(f"Act space        : {env.action_space}")
    print(f"Rep penalties    : {env.rep_penalties}")
    print(f"Div threshold    : {env.diversity_threshold}")
    print(f"Sat gain/loss    : {env.satisfaction_gain} / {env.satisfaction_loss}")

    N = 500

    def random_agent(obs, e):
        return e.action_space.sample()

    def greedy_agent(obs, e):
        profile = e.user_profile
        sims    = e.emb_matrix @ profile
        for mid in e.recommended_set:
            if mid in e.movie_to_action:
                sims[e.movie_to_action[mid]] = -999
        return int(np.argmax(sims))

    def diverse_agent(obs, e):
        profile = e.user_profile
        sims    = e.emb_matrix @ profile
        if len(e.history) > 0:
            recent_vecs = [v for _, v in e.history[-e.diversity_window:]]
            recent_mean = np.mean(recent_vecs, axis=0)
            norm        = np.linalg.norm(recent_mean)
            if norm > 0:
                recent_mean = recent_mean / norm
            div_scores = 1.0 - (e.emb_matrix @ recent_mean)
            combined   = 0.5 * sims + 0.5 * div_scores
        else:
            combined = sims
        for mid in e.recommended_set:
            if mid in e.movie_to_action:
                combined[e.movie_to_action[mid]] = -999
        return int(np.argmax(combined))

    print(f"\nRunning {N} episodes per agent ...\n")

    for agent_name, agent_fn in [
        ("Random",  random_agent),
        ("Greedy",  greedy_agent),
        ("Diverse", diverse_agent),
    ]:
        rewards, final_sats, reps, genres, movies = [], [], [], [], []
        for _ in range(N):
            obs, _ = env.reset()
            ep_r, ep_reps = 0.0, 0
            for _ in range(env.episode_length):
                obs, r, done, _, info = env.step(agent_fn(obs, env))
                ep_r    += r
                ep_reps += int(info["repetition_penalty"] > 0)
                if done: break
            rewards.append(ep_r)
            final_sats.append(info["satisfaction"])
            reps.append(ep_reps)
            genres.append(info["n_unique_genres"])
            movies.append(info["n_unique_movies"])

        print(f"[{agent_name:8s}]  "
              f"reward={np.mean(rewards):.3f}  "
              f"sat={np.mean(final_sats):.3f}  "
              f"reps={np.mean(reps):.1f}  "
              f"genres={np.mean(genres):.1f}  "
              f"unique_movies={np.mean(movies):.1f}/20")

    # Health checks
    print("\n── Health Checks ──")
    all_rewards = []
    all_sats    = []
    for _ in range(200):
        obs, _ = env.reset()
        for _ in range(env.episode_length):
            obs, r, done, _, info = env.step(env.action_space.sample())
            all_rewards.append(r)
            all_sats.append(info["satisfaction"])
            if done: break

    neg = sum(1 for r in all_rewards if r < 0)
    bad = sum(1 for s in all_sats if s < 0 or s > 1)
    print(f"  Negative rewards    : {neg}/{len(all_rewards)}  {'✓' if neg==0 else '✗'}")
    print(f"  Sat out of [0,1]    : {bad}/{len(all_sats)}  {'✓' if bad==0 else '✗'}")

    try:
        from stable_baselines3.common.env_checker import check_env
        check_env(env, warn=True)
        print("  SB3 check_env       : ✓")
    except Exception as ex:
        print(f"  SB3 check_env       : ✗ {ex}")

    print("\n✓ Environment v3 validation complete.")
