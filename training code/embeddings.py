# ============================================================
# STAGE 2 — NEURAL COLLABORATIVE FILTERING
# LEARN MOVIE EMBEDDINGS
# ============================================================
#
# GOAL:
# Train a neural collaborative filtering model and
# extract dense movie embeddings for downstream RL.
#
# INPUT:
# - processed_ratings.csv
#
# OUTPUT:
# - movie_embeddings.npy
# - user_embeddings.npy
# - ncf_model.pth
#
# ============================================================

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================

EMBED_DIM = 64
BATCH_SIZE = 8192
EPOCHS = 5
LR = 1e-3

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", DEVICE)

# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading processed ratings...")

df = pd.read_csv("C:\\Users\\deepa\\OneDrive\\Desktop\\RL project\\processed_ratings.csv")

print(df.head())

num_users = df["user_idx"].nunique()
num_movies = df["movie_idx"].nunique()

print("\nUsers :", num_users)
print("Movies:", num_movies)

# ============================================================
# NORMALIZE RATINGS
# ============================================================

# Scale ratings to 0-1

df["rating_norm"] = df["rating"] / 5.0

# ============================================================
# DATASET
# ============================================================

class RatingsDataset(Dataset):

    def __init__(self, dataframe):

        self.users = torch.LongTensor(
            dataframe["user_idx"].values
        )

        self.movies = torch.LongTensor(
            dataframe["movie_idx"].values
        )

        self.ratings = torch.FloatTensor(
            dataframe["rating_norm"].values
        )

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):

        return (
            self.users[idx],
            self.movies[idx],
            self.ratings[idx]
        )

dataset = RatingsDataset(df)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

# ============================================================
# NCF MODEL
# ============================================================

class NeuralCF(nn.Module):

    def __init__(
        self,
        num_users,
        num_movies,
        embed_dim=64
    ):
        super().__init__()

        # Embeddings
        self.user_embedding = nn.Embedding(
            num_users,
            embed_dim
        )

        self.movie_embedding = nn.Embedding(
            num_movies,
            embed_dim
        )

        # MLP
        self.mlp = nn.Sequential(

            nn.Linear(embed_dim * 2, 256),
            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 1),

            nn.Sigmoid()
        )

    def forward(self, users, movies):

        user_emb = self.user_embedding(users)
        movie_emb = self.movie_embedding(movies)

        x = torch.cat(
            [user_emb, movie_emb],
            dim=1
        )

        out = self.mlp(x)

        return out.squeeze()

# ============================================================
# INIT MODEL
# ============================================================

model = NeuralCF(
    num_users,
    num_movies,
    EMBED_DIM
).to(DEVICE)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR
)

# ============================================================
# TRAIN LOOP
# ============================================================

print("\nStarting NCF training...")

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    progress = tqdm(loader)

    for users, movies, ratings in progress:

        users = users.to(DEVICE)
        movies = movies.to(DEVICE)
        ratings = ratings.to(DEVICE)

        # Forward
        preds = model(users, movies)

        loss = criterion(preds, ratings)

        # Backprop
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

print("\nSaving model...")

torch.save(
    model.state_dict(),
    "C:\\Users\\deepa\\OneDrive\\Desktop\\RL project\\ncf_model.pth"
)

# ============================================================
# EXTRACT MOVIE EMBEDDINGS
# ============================================================

print("\nExtracting movie embeddings...")

movie_embeddings = (
    model.movie_embedding.weight
    .detach()
    .cpu()
    .numpy()
)

print("Movie embedding shape:", movie_embeddings.shape)

np.save(
    "C:\\Users\\deepa\\OneDrive\\Desktop\\RL project\\movie_embeddings.npy",
    movie_embeddings
)

# ============================================================
# EXTRACT USER EMBEDDINGS
# ============================================================

print("\nExtracting user embeddings...")

user_embeddings = (
    model.user_embedding.weight
    .detach()
    .cpu()
    .numpy()
)

print("User embedding shape:", user_embeddings.shape)

np.save(
    "C:\\Users\\deepa\\OneDrive\\Desktop\\RL project\\user_embeddings.npy",
    user_embeddings
)

# ============================================================
# DONE
# ============================================================

print("\n================================================")
print("NCF TRAINING COMPLETE")
print("Saved:")
print("- ncf_model.pth")
print("- movie_embeddings.npy")
print("- user_embeddings.npy")
print("================================================")