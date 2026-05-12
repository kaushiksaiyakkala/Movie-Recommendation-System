"""
Technical demo for the RL Movie Recommendation project.

What this script shows:
1. Start from one fixed MovieLens-style user history.
2. Run the same 20-step recommendation episode using Greedy, PPO, A2C, and DQN.
3. Print the movie each policy recommends at every timestep.
4. Compute reward, diversity, repetition, and engagement.
5. Save CSVs and plots for the presentation/demo.

Run:
    pip install -r requirements.txt
    python demo_episode.py
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from stable_baselines3 import A2C, DQN, PPO


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

def find_project_root(start: Path) -> Path:
    """Find the nearest folder at or above this file that contains data/ and models/."""
    for folder in [start, *start.parents]:
        if (folder / "data").exists() and (folder / "models").exists():
            return folder
    raise FileNotFoundError(
        "Could not find data/ and models/. Put app.py and demo_episode.py in your "
        "Movie-Recommendation-System folder, or copy data/ and models/ next to them."
    )

PROJECT_ROOT = find_project_root(BASE_DIR)
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "demo_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEQUENCE_LENGTH = 30
USER_STATE_DIM = 128
MOVIE_EMBED_DIM = 64
TOP_K = 100
EPISODE_LENGTH = 20

# Fixed seed for repeatable demo output.
np.random.seed(7)
torch.manual_seed(7)

# Edit this list if you want a different starting user profile.
# The script will fuzzy-match these names against MovieLens titles.
INITIAL_LIKED_MOVIES = [
    "Toy Story",
    "Finding Nemo",
    "Shrek",
    "Monsters, Inc.",
    "Incredibles, The",
]

POLICIES = ["greedy", "ppo", "a2c", "dqn"]


# ============================================================
# MODEL DEFINITIONS
# These match the training code architecture.
# ============================================================

class GRUStateEncoder(nn.Module):
    def __init__(self, movie_embeddings, embed_dim, hidden_dim, num_movies):
        super().__init__()
        self.movie_embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(movie_embeddings),
            freeze=True,
        )
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_movies),
        )

    def forward(self, sequences):
        x = self.movie_embedding(sequences)
        _, hidden = self.gru(x)
        hidden = hidden.squeeze(0)
        logits = self.output_layer(hidden)
        return logits, hidden


class RewardModel(nn.Module):
    def __init__(self, user_dim, movie_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(user_dim + movie_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, states, movies):
        x = torch.cat([states, movies], dim=1)
        return self.network(x).squeeze()


# ============================================================
# DEMO ENGINE
# ============================================================

class TechnicalMovieDemo:
    def __init__(self):
        print("Loading data/models...")
        print(f"Device: {DEVICE}")

        with open(DATA_DIR / "movie_id_map.pkl", "rb") as f:
            self.movie_encoder = pickle.load(f)

        self.movies_df = pd.read_csv(DATA_DIR / "movies.csv")
        self.movie_embeddings = np.load(DATA_DIR / "movie_embeddings_normalized.npy").astype("float32")
        self.num_movies = self.movie_embeddings.shape[0]
        self.faiss_index = faiss.read_index(str(DATA_DIR / "faiss_movie_index.bin"))

        self.movie_id_to_row = self.movies_df.set_index("movieId", drop=False)

        self.gru_model = GRUStateEncoder(
            self.movie_embeddings,
            MOVIE_EMBED_DIM,
            USER_STATE_DIM,
            self.num_movies,
        ).to(DEVICE)
        self.gru_model.load_state_dict(
            torch.load(MODEL_DIR / "gru_state_encoder.pth", map_location=DEVICE)
        )
        self.gru_model.eval()

        self.reward_model = RewardModel(USER_STATE_DIM, MOVIE_EMBED_DIM).to(DEVICE)
        self.reward_model.load_state_dict(
            torch.load(MODEL_DIR / "reward_model.pth", map_location=DEVICE)
        )
        self.reward_model.eval()

        # RL policies. These output an action index in the Top-100 candidate set.
        self.rl_models = {
            "ppo": PPO.load(str(MODEL_DIR / "ppo_movie_recommender"), device=DEVICE),
            "a2c": A2C.load(str(MODEL_DIR / "a2c_movie_recommender"), device=DEVICE),
            "dqn": DQN.load(str(MODEL_DIR / "dqn_movie_recommender"), device=DEVICE),
        }

    # --------------------------------------------------------
    # Movie ID/index helpers
    # --------------------------------------------------------

    def movie_idx_to_metadata(self, movie_idx: int) -> Dict[str, str]:
        movie_id = int(self.movie_encoder.inverse_transform([int(movie_idx)])[0])
        if movie_id in self.movie_id_to_row.index:
            row = self.movie_id_to_row.loc[movie_id]
            return {
                "movieId": movie_id,
                "title": str(row["title"]),
                "genres": str(row["genres"]),
            }
        return {
            "movieId": movie_id,
            "title": f"Movie index {movie_idx}",
            "genres": "unknown",
        }

    def title_to_movie_idx(self, query: str) -> int:
        query_lower = query.lower()
        matches = self.movies_df[
            self.movies_df["title"].str.lower().str.contains(query_lower, regex=False, na=False)
        ]
        if matches.empty:
            raise ValueError(f"Could not find a MovieLens title matching: {query}")

        # Prefer the earliest/cleanest match.
        movie_id = int(matches.iloc[0]["movieId"])
        return int(self.movie_encoder.transform([movie_id])[0])

    def build_initial_history(self, movie_queries: List[str]) -> List[int]:
        picked = []
        print("\nStarting user history:")
        for q in movie_queries:
            idx = self.title_to_movie_idx(q)
            meta = self.movie_idx_to_metadata(idx)
            picked.append(idx)
            print(f"  - {meta['title']} | {meta['genres']}")

        # GRU expects a sequence length of 30. For the demo, repeat the chosen
        # liked movies until we have enough history.
        history = []
        while len(history) < SEQUENCE_LENGTH:
            history.extend(picked)
        return history[-SEQUENCE_LENGTH:]

    # --------------------------------------------------------
    # State, candidates, reward
    # --------------------------------------------------------

    def get_state(self, history: List[int]) -> np.ndarray:
        seq = history[-SEQUENCE_LENGTH:]
        while len(seq) < SEQUENCE_LENGTH:
            seq.insert(0, seq[0])

        seq_tensor = torch.LongTensor(seq).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            _, hidden = self.gru_model(seq_tensor)
        return hidden.squeeze().cpu().numpy().astype(np.float32)

    def get_candidates(self, state: np.ndarray) -> np.ndarray:
        state_norm = state / (np.linalg.norm(state) + 1e-8)
        projected = state_norm[:MOVIE_EMBED_DIM].reshape(1, -1).astype("float32")
        faiss.normalize_L2(projected)
        _, indices = self.faiss_index.search(projected, TOP_K)
        return indices[0]

    def predicted_reward(self, state: np.ndarray, movie_idx: int) -> float:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        movie_tensor = torch.FloatTensor(self.movie_embeddings[int(movie_idx)]).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            return float(self.reward_model(state_tensor, movie_tensor).item())

    def choose_action(self, policy: str, state: np.ndarray, candidates: np.ndarray) -> int:
        if policy == "greedy":
            state_proj = state[:MOVIE_EMBED_DIM]
            sims = [float(np.dot(state_proj, self.movie_embeddings[int(idx)])) for idx in candidates]
            return int(np.argmax(sims))

        action, _ = self.rl_models[policy].predict(state, deterministic=True)
        return int(action)

    # --------------------------------------------------------
    # Run one 20-step episode for one policy
    # --------------------------------------------------------

    def run_episode(self, policy: str, initial_history: List[int]) -> pd.DataFrame:
        history = list(initial_history)
        rows = []
        recommended = []
        cumulative_reward = 0.0

        for t in range(1, EPISODE_LENGTH + 1):
            state = self.get_state(history)
            candidates = self.get_candidates(state)
            action = self.choose_action(policy, state, candidates)
            action = max(0, min(action, len(candidates) - 1))
            movie_idx = int(candidates[action])
            reward = self.predicted_reward(state, movie_idx)

            history.append(movie_idx)
            if len(history) > SEQUENCE_LENGTH:
                history.pop(0)

            recommended.append(movie_idx)
            cumulative_reward += reward

            diversity = len(set(recommended)) / len(recommended)
            repetition = 1.0 - diversity
            engagement = cumulative_reward * diversity
            meta = self.movie_idx_to_metadata(movie_idx)

            rows.append({
                "timestep": t,
                "policy": policy.upper(),
                "movie": meta["title"],
                "genres": meta["genres"],
                "reward": reward,
                "cumulative_reward": cumulative_reward,
                "diversity_so_far": diversity,
                "repetition_so_far": repetition,
                "engagement_so_far": engagement,
            })

        return pd.DataFrame(rows)

    # --------------------------------------------------------
    # Run all policies and save outputs
    # --------------------------------------------------------

    def run_all(self, initial_liked_movies: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        initial_history = self.build_initial_history(initial_liked_movies)

        all_episode_rows = []
        for policy in POLICIES:
            print(f"\nRunning 20-step episode for {policy.upper()}...")
            all_episode_rows.append(self.run_episode(policy, initial_history))

        episode_df = pd.concat(all_episode_rows, ignore_index=True)
        summary_df = self.summarize(episode_df)

        episode_path = OUTPUT_DIR / "episode_recommendations.csv"
        summary_path = OUTPUT_DIR / "final_metrics.csv"
        episode_df.to_csv(episode_path, index=False)
        summary_df.to_csv(summary_path, index=False)

        print(f"\nSaved episode table: {episode_path}")
        print(f"Saved metrics table: {summary_path}")

        self.save_plots(episode_df, summary_df)
        self.print_demo_tables(episode_df, summary_df)
        return episode_df, summary_df

    @staticmethod
    def summarize(episode_df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for policy, group in episode_df.groupby("policy"):
            final = group.iloc[-1]
            rows.append({
                "policy": policy,
                "final_reward": final["cumulative_reward"],
                "final_diversity": final["diversity_so_far"],
                "final_repetition": final["repetition_so_far"],
                "final_engagement": final["engagement_so_far"],
                "reward_stddev": group["reward"].std(ddof=0),
            })
        return pd.DataFrame(rows).sort_values("policy")

    def save_plots(self, episode_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
        # Cumulative reward over timesteps.
        plt.figure(figsize=(10, 6))
        for policy, group in episode_df.groupby("policy"):
            plt.plot(group["timestep"], group["cumulative_reward"], marker="o", label=policy)
        plt.xlabel("Timestep")
        plt.ylabel("Cumulative predicted reward")
        plt.title("Cumulative Reward During One Recommendation Episode")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "cumulative_reward_over_time.png", dpi=200)
        plt.close()

        # Diversity over timesteps.
        plt.figure(figsize=(10, 6))
        for policy, group in episode_df.groupby("policy"):
            plt.plot(group["timestep"], group["diversity_so_far"], marker="o", label=policy)
        plt.xlabel("Timestep")
        plt.ylabel("Title diversity so far")
        plt.title("Diversity During One Recommendation Episode")
        plt.ylim(0, 1.05)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "diversity_over_time.png", dpi=200)
        plt.close()

        # Final metrics bar chart.
        metric_cols = ["final_reward", "final_diversity", "final_repetition", "final_engagement"]
        for metric in metric_cols:
            plt.figure(figsize=(8, 5))
            plt.bar(summary_df["policy"], summary_df[metric])
            plt.xlabel("Policy")
            plt.ylabel(metric.replace("_", " "))
            plt.title(metric.replace("_", " ").title())
            plt.grid(True, axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / f"{metric}.png", dpi=200)
            plt.close()

        print(f"Saved plots in: {OUTPUT_DIR}")

    @staticmethod
    def print_demo_tables(episode_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
        print("\n================ 20-STEP RECOMMENDATION TABLE ================\n")
        pivot = episode_df.pivot(index="timestep", columns="policy", values="movie").reset_index()
        with pd.option_context("display.max_colwidth", 42, "display.width", 180):
            print(pivot.to_string(index=False))

        print("\n================ FINAL METRICS ================\n")
        pretty = summary_df.copy()
        for col in ["final_reward", "final_diversity", "final_repetition", "final_engagement", "reward_stddev"]:
            pretty[col] = pretty[col].map(lambda x: f"{x:.3f}")
        print(pretty.to_string(index=False))

        print("\nDemo line to say:")
        print("Greedy optimizes the immediate recommendation, while PPO/A2C/DQN are evaluated as sequential policies over a 20-step episode. The key comparison is not just raw reward, but reward plus diversity and repetition over time.")


def main():
    demo = TechnicalMovieDemo()
    demo.run_all(INITIAL_LIKED_MOVIES)


if __name__ == "__main__":
    main()
