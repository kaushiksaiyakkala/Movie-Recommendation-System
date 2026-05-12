# ============================================================
# utils/inference.py
# ============================================================

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pickle
import random
import numpy as np
import pandas as pd
import faiss

import torch
import torch.nn as nn

from stable_baselines3 import PPO, A2C, DQN

# ============================================================
# CONFIG
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SEQUENCE_LENGTH = 30

USER_STATE_DIM = 128
MOVIE_EMBED_DIM = 64

TOP_K = 100

# ============================================================
# MODEL DEFINITIONS
# ============================================================

class GRUStateEncoder(nn.Module):

    def __init__(
        self,
        movie_embeddings,
        embed_dim,
        hidden_dim,
        num_movies
    ):
        super().__init__()

        self.movie_embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(movie_embeddings),
            freeze=True
        )

        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.output_layer = nn.Sequential(

            nn.Linear(hidden_dim, 256),
            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(256, num_movies)
        )

    def forward(self, sequences):

        x = self.movie_embedding(sequences)

        _, hidden = self.gru(x)

        hidden = hidden.squeeze(0)

        logits = self.output_layer(hidden)

        return logits, hidden

# ------------------------------------------------------------

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

            nn.Linear(64, 1)
        )

    def forward(self, states, movies):

        x = torch.cat(
            [states, movies],
            dim=1
        )

        return self.network(x).squeeze()

# ============================================================
# MAIN RECOMMENDER
# ============================================================

class MovieRecommender:

    def __init__(self):

        # ====================================================
        # LOAD DATA
        # ====================================================

        with open("data/movie_id_map.pkl", "rb") as f:
            self.movie_encoder = pickle.load(f)

        self.movies_df = pd.read_csv(
            "data/movies.csv"
        )

        self.movie_embeddings = np.load(
            "data/movie_embeddings_normalized.npy"
        ).astype("float32")

        self.num_movies = self.movie_embeddings.shape[0]

        # ====================================================
        # LOAD FAISS
        # ====================================================

        self.faiss_index = faiss.read_index(
            "data/faiss_movie_index.bin"
        )

        # ====================================================
        # LOAD MODELS
        # ====================================================

        self.gru_model = GRUStateEncoder(
            self.movie_embeddings,
            MOVIE_EMBED_DIM,
            USER_STATE_DIM,
            self.num_movies
        ).to(DEVICE)

        self.gru_model.load_state_dict(
            torch.load(
                "models/gru_state_encoder.pth",
                map_location=DEVICE,
                weights_only=True
            )
        )

        self.gru_model.eval()

        # ----------------------------------------------------

        self.reward_model = RewardModel(
            USER_STATE_DIM,
            MOVIE_EMBED_DIM
        ).to(DEVICE)

        self.reward_model.load_state_dict(
            torch.load(
                "models/reward_model.pth",
                map_location=DEVICE,
                weights_only=True
            )
        )

        self.reward_model.eval()

        # ====================================================
        # LOAD RL MODELS
        # ====================================================

        self.ppo_model = PPO.load(
            "models/ppo_movie_recommender",
            device="cuda"
        )

        self.a2c_model = A2C.load(
            "models/a2c_movie_recommender",
            device="cuda"
        )

        self.dqn_model = DQN.load(
            "models/dqn_movie_recommender",
            device="cuda"
        )

        # ====================================================
        # STATE
        # ====================================================

        self.history = []

        self.reward_history = []

    # ========================================================
    # INITIALIZE USER
    # ========================================================

    def initialize_user(self, movie_titles):

        self.history = []

        for title in movie_titles:

            row = self.movies_df[
                self.movies_df["title"] == title
            ]

            if len(row) == 0:
                continue

            movie_id = row.iloc[0]["movieId"]

            movie_idx = self.movie_encoder.transform(
                [movie_id]
            )[0]

    # ========================================================
    # GET LATENT STATE
    # ========================================================

    def _get_state(self):

        if len(self.history) == 0:

            return np.zeros(
                USER_STATE_DIM,
                dtype=np.float32
            )

        padded_history = self.history[-SEQUENCE_LENGTH:]

        while len(padded_history) < SEQUENCE_LENGTH:

            padded_history.insert(0, padded_history[0])

        seq_tensor = torch.LongTensor(
            padded_history
        ).unsqueeze(0).to(DEVICE)

        with torch.no_grad():

            _, hidden = self.gru_model(seq_tensor)

        state = (
            hidden.squeeze()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        return state

    # ========================================================
    # CANDIDATE RETRIEVAL
    # ========================================================

    def _get_candidates(self, state):

        projected = state[:MOVIE_EMBED_DIM]

        projected = projected.reshape(1, -1).astype("float32")

        faiss.normalize_L2(projected)

        scores, indices = self.faiss_index.search(
            projected,
            TOP_K
        )

        return indices[0]

# ============================================================
# RECOMMEND
# UPDATED STABLE TOP-K RERANKING
# ============================================================

    def recommend(
        self,
        top_k=10,
        model_name="ppo",
        exploration_weight=0.3
    ):

        # ========================================================
        # GET CURRENT USER STATE
        # ========================================================

        state = self._get_state()

        # ========================================================
        # RETRIEVE CANDIDATES USING FAISS
        # ========================================================

        candidates = self._get_candidates(state)

        # ========================================================
        # SCORE ALL CANDIDATES
        # ========================================================

        candidate_scores = []

        state_tensor = torch.FloatTensor(
            state
        ).unsqueeze(0).to(DEVICE)

        for movie_idx in candidates:

            # ----------------------------------------------------
            # SKIP ALREADY WATCHED / RECOMMENDED
            # ----------------------------------------------------

            if movie_idx in self.history:
                continue

            # ----------------------------------------------------
            # MOVIE EMBEDDING
            # ----------------------------------------------------

            movie_emb = self.movie_embeddings[
                movie_idx
            ]

            movie_tensor = torch.FloatTensor(
                movie_emb
            ).unsqueeze(0).to(DEVICE)

            # ----------------------------------------------------
            # REWARD MODEL SCORE
            # ----------------------------------------------------

            with torch.no_grad():

                reward = self.reward_model(
                    state_tensor,
                    movie_tensor
                ).item()

            # ----------------------------------------------------
            # DIVERSITY BONUS
            # ----------------------------------------------------

            diversity = 0.0

            if len(self.history) > 0:

                recent_embs = self.movie_embeddings[
                    self.history[-10:]
                ]

                centroid = recent_embs.mean(axis=0)

                diversity = np.linalg.norm(
                    movie_emb - centroid
                )

            # ----------------------------------------------------
            # FINAL SCORE
            # ----------------------------------------------------

            final_score = (
                reward
                +
                exploration_weight * diversity
            )

            candidate_scores.append({

                "movie_idx": movie_idx,

                "reward": reward,

                "diversity": diversity,

                "score": final_score
            })

        # ========================================================
        # SORT BY FINAL SCORE
        # ========================================================

        candidate_scores = sorted(

            candidate_scores,

            key=lambda x: x["score"],

            reverse=True
        )

        # ========================================================
        # BUILD FINAL RECOMMENDATIONS
        # ========================================================

        recommendations = []

        for item in candidate_scores[:top_k]:

            movie_idx = item["movie_idx"]

            row = self.movies_df.iloc[
                movie_idx % len(self.movies_df)
            ]

            recommendations.append({

                "movie_idx": movie_idx,

                "title": row["title"],

                "genres": row["genres"],

                "reward": float(
                    item["reward"]
                ),

                "diversity": float(
                    item["diversity"]
                ),

                "score": float(
                    item["score"]
                )
            })

        return recommendations

    # ========================================================
    # UPDATE FEEDBACK
    # ========================================================

    def update_feedback(
        self,
        movie_idx,
        reward
    ):

        self.history.append(movie_idx)

        self.reward_history.append(reward)

    # ========================================================
    # RECENT HISTORY
    # ========================================================

    def get_recent_history(self):

        titles = []

        for idx in self.history[-10:]:

            row = self.movies_df.iloc[
                idx % len(self.movies_df)
            ]

            titles.append(row["title"])

        return titles

    # ========================================================
    # ANALYTICS
    # ========================================================

    def get_analytics(self):

        avg_reward = (
            np.mean(self.reward_history)
            if len(self.reward_history) > 0
            else 0
        )

        diversity = len(set(self.history)) / (
            len(self.history) + 1e-8
        )

        return {

            "total_recommendations":
                len(self.history),

            "average_reward":
                avg_reward,

            "diversity_score":
                diversity
        }

    # ========================================================
    # RESET
    # ========================================================

    def reset(self):

        self.history = []

        self.reward_history = []