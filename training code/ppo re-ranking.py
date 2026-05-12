# ============================================================
# STAGE 6 — PPO RL RE-RANKER
# ============================================================
#
# GOAL:
# Train PPO agent to:
#
# state -> choose best movie
#
# among retrieved candidates
#
# INPUT:
# - gru_state_encoder.pth
# - reward_model.pth
# - movie_embeddings.npy
# - user_sequences.pkl
# - faiss_movie_index.bin
#
# OUTPUT:
# - ppo_movie_recommender.zip
#
# ============================================================

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gymnasium as gym
from gymnasium import spaces

import faiss
import pickle
import random
import numpy as np

import torch
import torch.nn as nn

from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_checker import check_env

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

EPISODE_LENGTH = 20

# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading user sequences...")

with open("user_sequences.pkl", "rb") as f:
    user_sequences = pickle.load(f)

# ------------------------------------------------------------

print("\nLoading movie embeddings...")

movie_embeddings = np.load(
    "movie_embeddings_normalized.npy"
).astype("float32")

NUM_MOVIES = movie_embeddings.shape[0]

# ------------------------------------------------------------

print("\nLoading FAISS index...")

faiss_index = faiss.read_index(
    "faiss_movie_index.bin"
)

# ============================================================
# LOAD GRU ENCODER
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

print("\nLoading GRU encoder...")

gru_model = GRUStateEncoder(
    movie_embeddings,
    MOVIE_EMBED_DIM,
    USER_STATE_DIM,
    NUM_MOVIES
).to(DEVICE)

gru_model.load_state_dict(
    torch.load(
        "gru_state_encoder.pth",
        map_location=DEVICE
    )
)

gru_model.eval()

# ============================================================
# LOAD REWARD MODEL
# ============================================================

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

# ------------------------------------------------------------

print("\nLoading reward model...")

reward_model = RewardModel(
    USER_STATE_DIM,
    MOVIE_EMBED_DIM
).to(DEVICE)

reward_model.load_state_dict(
        torch.load(
            "reward_model.pth",
            map_location=DEVICE,
            weights_only=True
    )
)

reward_model.eval()

# ============================================================
# CUSTOM RL ENVIRONMENT
# ============================================================

class MovieRecommendationEnv(gym.Env):

    def __init__(self):

        super().__init__()

        # action:
        # choose among TOP_K candidates
        self.action_space = spaces.Discrete(
            TOP_K
        )

        # state:
        # latent user state
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(USER_STATE_DIM,),
            dtype=np.float32
        )

    # --------------------------------------------------------

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        # random user
        self.user_id = random.choice(
            list(user_sequences.keys())
        )

        self.sequence = user_sequences[self.user_id]

        # initial history
        self.current_history = [
            x["movie_idx"]
            for x in self.sequence[:SEQUENCE_LENGTH]
        ]

        self.step_count = 0

        state = self._get_state()

        return state, {}

    # --------------------------------------------------------

    def _get_state(self):

        seq_tensor = torch.LongTensor(
            self.current_history[-SEQUENCE_LENGTH:]
        ).unsqueeze(0).to(DEVICE)

        with torch.no_grad():

            _, hidden = gru_model(seq_tensor)

        state = (
            hidden.squeeze()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        return state

    # --------------------------------------------------------

    def _get_candidates(self, state):

        state_norm = state / (
            np.linalg.norm(state) + 1e-8
        )

        # project user state to movie embedding dim
        projected = state_norm[:MOVIE_EMBED_DIM]

        projected = projected.reshape(1, -1).astype("float32")

        faiss.normalize_L2(projected)

        scores, indices = faiss_index.search(
            projected,
            TOP_K
        )

        return indices[0]

    # --------------------------------------------------------

    def step(self, action):

        state = self._get_state()

        candidates = self._get_candidates(state)

        movie_idx = candidates[action]

        movie_emb = movie_embeddings[movie_idx]

        # reward prediction
        state_tensor = torch.FloatTensor(
            state
        ).unsqueeze(0).to(DEVICE)

        movie_tensor = torch.FloatTensor(
            movie_emb
        ).unsqueeze(0).to(DEVICE)

        with torch.no_grad():

            reward = reward_model(
                state_tensor,
                movie_tensor
            ).item()

        # update history
        self.current_history.append(movie_idx)

        if len(self.current_history) > SEQUENCE_LENGTH:
            self.current_history.pop(0)

        self.step_count += 1

        terminated = (
            self.step_count >= EPISODE_LENGTH
        )

        next_state = self._get_state()

        info = {}

        return (
            next_state,
            reward,
            terminated,
            False,
            info
        )

# ============================================================
# CREATE ENV
# ============================================================

env = MovieRecommendationEnv()

check_env(env)

print("\nEnvironment check passed.")

# ============================================================
# PPO MODEL
# ============================================================

print("\nInitializing PPO...")

model = PPO(

    policy="MlpPolicy",

    env=env,

    verbose=1,

    learning_rate=3e-4,

    batch_size=64,

    gamma=0.99,

    ent_coef=0.01,

    n_steps=2048,

    tensorboard_log="./ppo_logs/",

    device="cuda"
)

# ============================================================
# TRAIN PPO
# ============================================================

print("\nStarting PPO training...")

model.learn(

    total_timesteps=100_000,

    progress_bar=True
)

# ============================================================
# SAVE MODEL
# ============================================================

print("\nSaving PPO model...")

model.save(
    "ppo_movie_recommender"
)

# ============================================================
# DONE
# ============================================================

print("\n================================================")
print("PPO TRAINING COMPLETE")
print("Saved:")
print("- ppo_movie_recommender.zip")
print("================================================")


# ============================================================
# A2C
# ============================================================

print("\nInitializing A2C...")

a2c_model = A2C(

    policy="MlpPolicy",

    env=env,

    verbose=1,

    learning_rate=7e-4,

    gamma=0.99,

    tensorboard_log="./a2c_logs/",

    device="cuda"
)

print("\nTraining A2C...")

a2c_model.learn(
    total_timesteps=50000,
    progress_bar=True
)

a2c_model.save(
    "a2c_movie_recommender"
)

print("\nA2C complete.")

# ============================================================
# DQN
# ============================================================

print("\nInitializing DQN...")

dqn_model = DQN(

    policy="MlpPolicy",

    env=env,

    verbose=1,

    learning_rate=1e-4,

    batch_size=64,

    gamma=0.99,

    buffer_size=50000,

    learning_starts=5000,

    target_update_interval=1000,

    tensorboard_log="./dqn_logs/",

    device="cuda"
)

print("\nTraining DQN...")

dqn_model.learn(
    total_timesteps=50000,
    progress_bar=True
)

dqn_model.save(
    "dqn_movie_recommender"
)

print("\nDQN complete.")