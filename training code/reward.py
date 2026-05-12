# ============================================================
# STAGE 4 — NEURAL REWARD MODEL
# ============================================================
#
# GOAL:
# Learn a reward function:
#
# R(state, movie) -> expected reward
#
# This approximates:
# - user satisfaction
# - relevance
# - diversity
# - novelty
#
# INPUT:
# - user_sequences.pkl
# - user_state_embeddings.pkl
# - movie_embeddings.npy
#
# OUTPUT:
# - reward_model.pth
#
# ============================================================

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pickle
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ============================================================
# CONFIG
# ============================================================

MOVIE_EMBED_DIM = 64
USER_STATE_DIM = 128

BATCH_SIZE = 4096
EPOCHS = 3
LR = 1e-3

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", DEVICE)

# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading user sequences...")

with open("C:\\Users\\deepa\\OneDrive\\Desktop\\RL project\\user_sequences.pkl", "rb") as f:
    user_sequences = pickle.load(f)

print("Loaded user sequences.")

# ------------------------------------------------------------

print("\nLoading user state embeddings...")

with open("C:\\Users\\deepa\\OneDrive\\Desktop\\RL project\\user_state_embeddings.pkl", "rb") as f:
    user_state_embeddings = pickle.load(f)

print("Loaded user states.")

# ------------------------------------------------------------

print("\nLoading movie embeddings...")

movie_embeddings = np.load("C:\\Users\\deepa\\OneDrive\\Desktop\\RL project\\movie_embeddings.npy")

print("Movie embedding shape:", movie_embeddings.shape)

# ============================================================
# BUILD TRAINING DATA
# ============================================================

print("\nBuilding reward training samples...")

X_states = []
X_movies = []
Y_rewards = []

for user_id, seq in tqdm(user_sequences.items()):

    # skip users without state embeddings
    if user_id not in user_state_embeddings:
        continue

    user_state = user_state_embeddings[user_id]

    recent_movies = []

    for interaction in seq:

        movie_idx = interaction["movie_idx"]
        rating = interaction["rating"]

        movie_emb = movie_embeddings[movie_idx]

        # ====================================================
        # DIVERSITY SCORE
        # ====================================================

        diversity_bonus = 0.0

        if len(recent_movies) > 0:

            recent_embs = movie_embeddings[recent_movies]

            centroid = recent_embs.mean(axis=0)

            distance = np.linalg.norm(
                movie_emb - centroid
            )

            diversity_bonus = distance * 0.05

        # ====================================================
        # FINAL REWARD
        # ====================================================

        reward = (
            0.9 * (rating / 5.0)
            + 0.1 * diversity_bonus
        )

        X_states.append(user_state)
        X_movies.append(movie_emb)
        Y_rewards.append(reward)

        # update history
        recent_movies.append(movie_idx)

        if len(recent_movies) > 10:
            recent_movies.pop(0)

# ============================================================
# CONVERT TO TENSORS
# ============================================================

X_states = torch.FloatTensor(
    np.array(X_states)
)

X_movies = torch.FloatTensor(
    np.array(X_movies)
)

Y_rewards = torch.FloatTensor(
    np.array(Y_rewards)
)

print("\nTraining samples:", len(Y_rewards))

# ============================================================
# DATASET
# ============================================================

class RewardDataset(Dataset):

    def __init__(
        self,
        states,
        movies,
        rewards
    ):

        self.states = states
        self.movies = movies
        self.rewards = rewards

    def __len__(self):
        return len(self.rewards)

    def __getitem__(self, idx):

        return (
            self.states[idx],
            self.movies[idx],
            self.rewards[idx]
        )

dataset = RewardDataset(
    X_states,
    X_movies,
    Y_rewards
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

# ============================================================
# REWARD MODEL
# ============================================================

class RewardModel(nn.Module):

    def __init__(
        self,
        user_dim,
        movie_dim
    ):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                user_dim + movie_dim,
                256
            ),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(256, 128),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(128, 64),

            nn.ReLU(),

            nn.Linear(64, 1)
        )

    def forward(
        self,
        states,
        movies
    ):

        x = torch.cat(
            [states, movies],
            dim=1
        )

        return self.network(x).squeeze()

# ============================================================
# INIT MODEL
# ============================================================

model = RewardModel(
    USER_STATE_DIM,
    MOVIE_EMBED_DIM
).to(DEVICE)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR
)

# ============================================================
# TRAIN LOOP
# ============================================================

print("\nStarting reward model training...")

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    progress = tqdm(loader)

    for states, movies, rewards in progress:

        states = states.to(DEVICE)
        movies = movies.to(DEVICE)
        rewards = rewards.to(DEVICE)

        # forward
        preds = model(states, movies)

        loss = criterion(
            preds,
            rewards
        )

        # backward
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        progress.set_description(
            f"Epoch {epoch+1}/{EPOCHS} "
            f"Loss: {loss.item():.4f}"
        )

    avg_loss = total_loss / len(loader)

    print(
        f"\nEpoch {epoch+1} complete "
        f"| Avg Loss: {avg_loss:.4f}"
    )

# ============================================================
# SAVE MODEL
# ============================================================

print("\nSaving reward model...")

torch.save(
    model.state_dict(),
    "reward_model.pth"
)

# ============================================================
# DONE
# ============================================================

print("\n================================================")
print("REWARD MODEL TRAINING COMPLETE")
print("Saved:")
print("- reward_model.pth")
print("================================================")