"""
environment.py
==============
Custom Gymnasium environment for RL-based movie recommendation.

State (65-dim):
  - 64-dim: mean BPR embedding of last K recommended movies
  - 1-dim:  current satisfaction score [0, 1]

Action:
  - Integer 0-499: index into catalog

Episode:
  - Fixed length of 20 steps
  - New random user sampled each episode

Satisfaction targets:
  - Greedy agent:  final sat ~0.0  (collapses from repetition)
  - Random agent:  final sat ~0.4-0.6
  - Good RL agent: final sat ~0.7-0.9
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
        satisfaction_gain=0.06,
        satisfaction_loss=0.10,
        diversity_threshold=0.42,
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

        self.reward_weights = reward_weights or {
            "rating":     0.6,
            "diversity":  0.3,
            "repetition": 0.1,
        }

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
        self.current_user_id  = None
        self.user_profile     = None
        self.user_rating_dict = {}
        self.satisfaction     = self.initial_satisfaction
        self.timestep         = 0
        self.history          = []
        self.recommended_set  = set()
        self.genres_seen      = set()

        self.rng = np.random.default_rng(seed)

    # ────────────────────────────────────────────────────────────────────
    # reset
    # ────────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_user_id  = int(self.rng.choice(self.user_ids))
        self.user_profile     = self.user_profiles[self.current_user_id]
        self.user_rating_dict = self.user_ratings.get(self.current_user_id, {})
        self.satisfaction     = self.initial_satisfaction
        self.timestep         = 0
        self.history          = []
        self.recommended_set  = set()
        self.genres_seen      = set()

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

        # ── 2. Diversity ─────────────────────────────────────────────────
        emb_div         = self._embedding_diversity(movie_vec)
        genre_div       = self._genre_diversity(movie_id)
        diversity_bonus = float(np.clip(0.7 * emb_div + 0.3 * genre_div, 0.0, 1.0))

        # ── 3. Repetition ────────────────────────────────────────────────
        repetition_penalty = 1.0 if movie_id in self.recommended_set else 0.0

        # ── 4. Reward ────────────────────────────────────────────────────
        w = self.reward_weights
        raw = max(0.0,
            w["rating"]     * rating_score
          + w["diversity"]  * diversity_bonus
          - w["repetition"] * repetition_penalty
        )
        reward = float(raw * self.satisfaction)

        # ── 5. Satisfaction ──────────────────────────────────────────────
        if diversity_bonus >= self.diversity_threshold:
            self.satisfaction = min(1.0, self.satisfaction + self.satisfaction_gain)
        else:
            self.satisfaction = max(0.0, self.satisfaction - self.satisfaction_loss)

        # ── 6. Update ────────────────────────────────────────────────────
        self.history.append((movie_id, movie_vec))
        self.recommended_set.add(movie_id)
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
# Comprehensive validation
# ────────────────────────────────────────────────────────────────────────

def run_agent(env, policy_fn, n_episodes, desc=""):
    """
    Run an agent for n_episodes. Returns dict of collected metrics.
    policy_fn(obs, env) -> action
    """
    rewards        = []
    final_sats     = []
    sat_curves     = []   # satisfaction at each timestep averaged across eps
    div_curves     = []
    rating_curves  = []
    unique_genres  = []
    unique_movies  = []
    rep_counts     = []   # total repetitions per episode

    sat_per_step   = np.zeros(env.episode_length)
    div_per_step   = np.zeros(env.episode_length)
    rat_per_step   = np.zeros(env.episode_length)
    count_per_step = np.zeros(env.episode_length)

    for ep in range(n_episodes):
        obs, info = env.reset()
        ep_reward  = 0.0
        ep_reps    = 0
        ep_sat     = []
        ep_div     = []
        ep_rat     = []

        for t in range(env.episode_length):
            action = policy_fn(obs, env)
            obs, reward, terminated, _, info = env.step(action)
            ep_reward += reward
            ep_reps   += int(info["repetition_penalty"])
            ep_sat.append(info["satisfaction"])
            ep_div.append(info["diversity_bonus"])
            ep_rat.append(info["rating_score"])

            sat_per_step[t]   += info["satisfaction"]
            div_per_step[t]   += info["diversity_bonus"]
            rat_per_step[t]   += info["rating_score"]
            count_per_step[t] += 1

            if terminated:
                break

        rewards.append(ep_reward)
        final_sats.append(ep_sat[-1])
        unique_genres.append(info["n_unique_genres"])
        unique_movies.append(info["n_unique_movies"])
        rep_counts.append(ep_reps)

    # Average curves
    for t in range(env.episode_length):
        if count_per_step[t] > 0:
            sat_curves.append(sat_per_step[t] / count_per_step[t])
            div_curves.append(div_per_step[t] / count_per_step[t])
            rating_curves.append(rat_per_step[t] / count_per_step[t])

    return {
        "name":           desc,
        "n_episodes":     n_episodes,
        "rewards":        rewards,
        "final_sats":     final_sats,
        "unique_genres":  unique_genres,
        "unique_movies":  unique_movies,
        "rep_counts":     rep_counts,
        "sat_curve":      sat_curves,
        "div_curve":      div_curves,
        "rating_curve":   rating_curves,
    }


def print_results(res):
    r  = np.array(res["rewards"])
    s  = np.array(res["final_sats"])
    g  = np.array(res["unique_genres"])
    m  = np.array(res["unique_movies"])
    rp = np.array(res["rep_counts"])

    print(f"\n  [{res['name']}]  ({res['n_episodes']} episodes)")
    print(f"  Cumulative reward    : {r.mean():.3f} ± {r.std():.3f}  "
          f"[min={r.min():.2f}  max={r.max():.2f}]")
    print(f"  Final satisfaction   : {s.mean():.3f} ± {s.std():.3f}  "
          f"[min={s.min():.2f}  max={s.max():.2f}]")
    print(f"  Unique genres/ep     : {g.mean():.1f} ± {g.std():.1f}")
    print(f"  Unique movies/ep     : {m.mean():.1f} ± {m.std():.1f}  (out of 20)")
    print(f"  Repetitions/ep       : {rp.mean():.2f} ± {rp.std():.2f}")
    print(f"  % eps with rep > 0   : {100*np.mean(rp > 0):.1f}%")

    # Satisfaction curve (summarized)
    curve = res["sat_curve"]
    early = np.mean(curve[:5])    if len(curve) >= 5  else 0
    mid   = np.mean(curve[8:12])  if len(curve) >= 12 else 0
    late  = np.mean(curve[15:])   if len(curve) >= 16 else 0
    print(f"  Sat curve (early/mid/late): {early:.3f} / {mid:.3f} / {late:.3f}")


def compare_agents(results_list):
    """Print side-by-side comparison."""
    print("\n" + "="*65)
    print("  AGENT COMPARISON SUMMARY")
    print("="*65)
    print(f"  {'Agent':20s} {'Reward':>10s} {'Final Sat':>10s} "
          f"{'Genres':>8s} {'Reps':>6s}")
    print("  " + "-"*60)
    for res in results_list:
        r  = np.mean(res["rewards"])
        s  = np.mean(res["final_sats"])
        g  = np.mean(res["unique_genres"])
        rp = np.mean(res["rep_counts"])
        print(f"  {res['name']:20s} {r:>10.3f} {s:>10.3f} {g:>8.1f} {rp:>6.2f}")
    print("="*65)


if __name__ == "__main__":
    import sys

    print("=" * 65)
    print("  ENVIRONMENT COMPREHENSIVE VALIDATION")
    print("=" * 65)

    env = MovieRecEnv(data_dir="./data/processed")

    print(f"\n  Obs space         : {env.observation_space}")
    print(f"  Act space         : {env.action_space}")
    print(f"  Users             : {len(env.user_ids)}")
    print(f"  Catalog           : {env.n_movies} movies")
    print(f"  Embed dim         : {env.embed_dim}")
    print(f"  Episode length    : {env.episode_length}")
    print(f"  Diversity threshold: {env.diversity_threshold}")
    print(f"  Satisfaction gain : {env.satisfaction_gain}")
    print(f"  Satisfaction loss : {env.satisfaction_loss}")

    N = 500  # episodes per agent — statistically meaningful

    # ── Define agents ────────────────────────────────────────────────────
    def random_agent(obs, env):
        return env.action_space.sample()

    def greedy_agent(obs, env):
        """Always picks highest BPR similarity to user profile, no diversity."""
        profile = env.user_profile
        sims    = env.emb_matrix @ profile
        # Mask already recommended movies
        for mid in env.recommended_set:
            if mid in env.movie_to_action:
                sims[env.movie_to_action[mid]] = -999
        return int(np.argmax(sims))

    def diverse_agent(obs, env):
        """
        Picks highest BPR similarity BUT penalises movies similar to recent recs.
        Approximates what a good RL agent should learn.
        """
        profile = env.user_profile
        sims    = env.emb_matrix @ profile  # (500,) rating signal

        if len(env.history) > 0:
            recent_vecs = [v for _, v in env.history[-env.diversity_window:]]
            recent_mean = np.mean(recent_vecs, axis=0)
            norm        = np.linalg.norm(recent_mean)
            if norm > 0:
                recent_mean = recent_mean / norm
            div_scores = 1.0 - (env.emb_matrix @ recent_mean)  # (500,)
            combined   = 0.5 * sims + 0.5 * div_scores
        else:
            combined = sims

        # Mask already recommended
        for mid in env.recommended_set:
            if mid in env.movie_to_action:
                combined[env.movie_to_action[mid]] = -999

        return int(np.argmax(combined))

    def topk_agent(obs, env):
        """
        Picks randomly from top-20 BPR matches — explores within taste.
        """
        profile = env.user_profile
        sims    = env.emb_matrix @ profile
        for mid in env.recommended_set:
            if mid in env.movie_to_action:
                sims[env.movie_to_action[mid]] = -999
        top20 = np.argsort(sims)[-20:]
        return int(env.rng.choice(top20))

    # ── Run all agents ───────────────────────────────────────────────────
    print(f"\n  Running {N} episodes per agent ...\n")

    results = []
    for name, fn in [
        ("Random",        random_agent),
        ("Greedy",        greedy_agent),
        ("Diverse",       diverse_agent),
        ("TopK-Random",   topk_agent),
    ]:
        print(f"  Running {name} ...")
        res = run_agent(env, fn, N, desc=name)
        print_results(res)
        results.append(res)

    compare_agents(results)

    # ── Reward signal validation ─────────────────────────────────────────
    print("\n=== Reward Signal Validation (500 episodes, random agent) ===")
    rated_rewards   = []
    unrated_rewards = []

    for _ in range(500):
        obs, info = env.reset()
        for t in range(env.episode_length):
            action = random_agent(obs, env)
            movie_id = env.catalog[action]
            obs, reward, terminated, _, info = env.step(action)
            if movie_id in env.user_rating_dict:
                rated_rewards.append(info["rating_score"])
            else:
                unrated_rewards.append(info["rating_score"])
            if terminated:
                break

    print(f"  Rated movies   — mean rating score : {np.mean(rated_rewards):.3f} "
          f"± {np.std(rated_rewards):.3f}  (n={len(rated_rewards):,})")
    print(f"  Unrated movies — mean rating score : {np.mean(unrated_rewards):.3f} "
          f"± {np.std(unrated_rewards):.3f}  (n={len(unrated_rewards):,})")
    print(f"  Rated vs unrated gap: {np.mean(rated_rewards)-np.mean(unrated_rewards):.3f}"
          f"  (positive = rated movies score higher on avg)")

    # ── Diversity distribution ───────────────────────────────────────────
    print("\n=== Diversity Distribution (500 random episodes) ===")
    all_divs = []
    obs, _ = env.reset()
    for _ in range(500):
        obs, _ = env.reset()
        for t in range(env.episode_length):
            action = random_agent(obs, env)
            obs, _, terminated, _, info = env.step(action)
            all_divs.append(info["diversity_bonus"])
            if terminated:
                break

    all_divs = np.array(all_divs)
    print(f"  Mean diversity bonus : {all_divs.mean():.3f}")
    print(f"  Std diversity bonus  : {all_divs.std():.3f}")
    print(f"  % above threshold ({env.diversity_threshold}) : "
          f"{100*np.mean(all_divs >= env.diversity_threshold):.1f}%  "
          f"(want ~45-55% for random)")
    buckets = [0, 0.2, 0.4, 0.6, 0.8, 1.01]
    print(f"  Distribution:")
    for i in range(len(buckets)-1):
        pct = 100 * np.mean((all_divs >= buckets[i]) & (all_divs < buckets[i+1]))
        bar = "█" * int(pct / 2)
        print(f"    [{buckets[i]:.1f}-{buckets[i+1]:.1f}): {pct:5.1f}%  {bar}")

    # ── Satisfaction curve comparison ────────────────────────────────────
    print("\n=== Satisfaction Curves (timestep by timestep) ===")
    print(f"  {'Step':>4s}  {'Random':>8s}  {'Greedy':>8s}  "
          f"{'Diverse':>8s}  {'TopK':>8s}")
    print("  " + "-"*45)
    for t in range(env.episode_length):
        row = f"  {t+1:>4d}"
        for res in results:
            if t < len(res["sat_curve"]):
                row += f"  {res['sat_curve'][t]:>8.3f}"
            else:
                row += f"  {'N/A':>8s}"
        print(row)

    # ── Environment health checks ────────────────────────────────────────
    print("\n=== Environment Health Checks ===")

    # Check 1: obs shape
    obs, _ = env.reset()
    expected = (env.embed_dim + 1,)
    all_ok = True
    for _ in range(env.episode_length):
        obs, _, done, _, _ = env.step(env.action_space.sample())
        if obs.shape != expected:
            print(f"  ✗ Bad obs shape: {obs.shape}")
            all_ok = False
            break
        if done:
            break
    if all_ok:
        print(f"  ✓ Obs shape consistent: {expected}")

    # Check 2: obs bounds
    obs_samples = []
    for _ in range(100):
        obs, _ = env.reset()
        obs_samples.append(obs)
        for _ in range(env.episode_length):
            obs, _, done, _, _ = env.step(env.action_space.sample())
            obs_samples.append(obs)
            if done: break
    obs_arr = np.array(obs_samples)
    print(f"  ✓ Obs range: [{obs_arr.min():.3f}, {obs_arr.max():.3f}]  "
          f"(space: [{env.observation_space.low.min():.1f}, "
          f"{env.observation_space.high.max():.1f}])")

    # Check 3: reward always >= 0
    all_rewards = []
    for _ in range(200):
        obs, _ = env.reset()
        for _ in range(env.episode_length):
            obs, r, done, _, _ = env.step(env.action_space.sample())
            all_rewards.append(r)
            if done: break
    neg = sum(1 for r in all_rewards if r < 0)
    print(f"  {'✓' if neg == 0 else '✗'} Negative rewards: {neg}/{len(all_rewards)}")

    # Check 4: satisfaction always in [0,1]
    all_sats_check = []
    for _ in range(200):
        obs, _ = env.reset()
        for _ in range(env.episode_length):
            obs, _, done, _, info = env.step(env.action_space.sample())
            all_sats_check.append(info["satisfaction"])
            if done: break
    bad_sat = sum(1 for s in all_sats_check if s < 0 or s > 1)
    print(f"  {'✓' if bad_sat == 0 else '✗'} Satisfaction bounds [0,1]: "
          f"{bad_sat} violations")

    # Check 5: SB3 compatibility
    try:
        from stable_baselines3.common.env_checker import check_env
        check_env(env, warn=True)
        print("  ✓ SB3 check_env passed")
    except Exception as e:
        print(f"  ✗ SB3 check_env: {e}")

    # ── Final verdict ────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  FINAL VERDICT")
    print("="*65)

    greedy_res  = results[1]
    diverse_res = results[2]
    random_res  = results[0]

    greedy_reward  = np.mean(greedy_res["rewards"])
    diverse_reward = np.mean(diverse_res["rewards"])
    random_reward  = np.mean(random_res["rewards"])
    greedy_sat     = np.mean(greedy_res["final_sats"])
    random_sat     = np.mean(random_res["final_sats"])
    diverse_sat    = np.mean(diverse_res["final_sats"])

    checks = {
        "Greedy reward < Random reward"  : greedy_reward  < random_reward,
        "Diverse reward > Random reward" : diverse_reward > random_reward,
        "Greedy sat < 0.3"               : greedy_sat     < 0.3,
        "Random sat in [0.3, 0.75]"      : 0.3 <= random_sat <= 0.75,
        "Diverse sat > Random sat"        : diverse_sat    > random_sat,
        "Reward gap > 2.0"               : (diverse_reward - greedy_reward) > 2.0,
        "No negative rewards"            : neg == 0,
        "Satisfaction in [0,1]"          : bad_sat == 0,
    }

    all_pass = True
    for check, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"  {status}  {check}")
        if not passed:
            all_pass = False

    print("\n  " + ("✓ ALL CHECKS PASSED — environment ready for training"
                    if all_pass else
                    "✗ SOME CHECKS FAILED — review parameters above"))
    print("="*65)
