# ============================================================
# STAGE 3 — GRU USER STATE ENCODER
# ============================================================
#
# GOAL:
# Learn latent user state representations from
# sequential movie interaction histories.
#
# INPUT:
# - user_sequences.pkl
# - movie_embeddings.npy
#
# OUTPUT:
# - gru_state_encoder.pth
#
# WHAT THIS LEARNS:
# - evolving user taste
# - genre drift
# - sequential preference patterns
# - latent satisfaction dynamics
#
# ============================================================

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pickle
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ============================================================
# CONFIG
# ============================================================

SEQUENCE_LENGTH = 30

EMBED_DIM = 64
HIDDEN_DIM = 128

BATCH_SIZE = 2048
EPOCHS = 5
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

print("Users loaded:", len(user_sequences))

print("\nLoading movie embeddings...")

movie_embeddings = np.load("C:\\Users\\deepa\\OneDrive\\Desktop\\RL project\\movie_embeddings.npy")

print("Movie embeddings shape:", movie_embeddings.shape)

movie_embeddings_tensor = torch.FloatTensor(
    movie_embeddings
)

NUM_MOVIES = movie_embeddings.shape[0]

# ============================================================
# BUILD TRAINING SAMPLES
# ============================================================

print("\nBuilding sequential samples...")

samples = []

for user_id, seq in tqdm(user_sequences.items()):

    if len(seq) < SEQUENCE_LENGTH + 1:
        continue

    movie_seq = [x["movie_idx"] for x in seq]
    liked_seq = [x["liked"] for x in seq]

    # sliding window
    for i in range(SEQUENCE_LENGTH, len(movie_seq)):

        input_seq = movie_seq[i-SEQUENCE_LENGTH:i]

        target_movie = movie_seq[i]

        target_liked = liked_seq[i]

        # ONLY learn from positive interactions
        if target_liked == 1:

            samples.append((
                input_seq,
                target_movie
            ))

print("Total training samples:", len(samples))

# ============================================================
# DATASET
# ============================================================

class SequentialDataset(Dataset):

    def __init__(self, samples):

        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        seq, target = self.samples[idx]

        return (
            torch.LongTensor(seq),
            torch.LongTensor([target])
        )

dataset = SequentialDataset(samples)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

# ============================================================
# GRU STATE ENCODER
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

        # pretrained movie embeddings
        self.movie_embedding = nn.Embedding.from_pretrained(
            movie_embeddings,
            freeze=False
        )

        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        # next-movie prediction head
        self.output_layer = nn.Sequential(

            nn.Linear(hidden_dim, 256),
            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(256, num_movies)
        )

    def forward(self, sequences):

        # sequence embeddings
        x = self.movie_embedding(sequences)

        # GRU
        _, hidden = self.gru(x)

        # hidden shape:
        # [1, batch, hidden_dim]

        hidden = hidden.squeeze(0)

        logits = self.output_layer(hidden)

        return logits, hidden

# ============================================================
# INIT MODEL
# ============================================================

model = GRUStateEncoder(
    movie_embeddings_tensor,
    EMBED_DIM,
    HIDDEN_DIM,
    NUM_MOVIES
).to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR
)

# ============================================================
# TRAIN LOOP
# ============================================================

print("\nStarting GRU training...")

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    progress = tqdm(loader)

    for sequences, targets in progress:

        sequences = sequences.to(DEVICE)

        targets = targets.squeeze().to(DEVICE)

        # forward
        logits, hidden = model(sequences)

        loss = criterion(logits, targets)

        # backward
        optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            5.0
        )

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

print("\nSaving GRU encoder...")

torch.save(
    model.state_dict(),
    "gru_state_encoder.pth"
)

# ============================================================
# OPTIONAL:
# SAVE USER STATES
# ============================================================

print("\nGenerating final user state embeddings...")

model.eval()

user_state_embeddings = {}

with torch.no_grad():

    for user_id, seq in tqdm(user_sequences.items()):

        if len(seq) < SEQUENCE_LENGTH:
            continue

        movie_seq = [
            x["movie_idx"]
            for x in seq[-SEQUENCE_LENGTH:]
        ]

        movie_seq_tensor = torch.LongTensor(
            movie_seq
        ).unsqueeze(0).to(DEVICE)

        _, hidden = model(movie_seq_tensor)

        user_state_embeddings[user_id] = (
            hidden.squeeze()
            .cpu()
            .numpy()
        )

# ============================================================
# SAVE USER STATES
# ============================================================

with open("user_state_embeddings.pkl", "wb") as f:

    pickle.dump(
        user_state_embeddings,
        f
    )

# ============================================================
# DONE
# ============================================================

print("\n================================================")
print("GRU STATE ENCODER TRAINING COMPLETE")
print("Saved:")
print("- gru_state_encoder.pth")
print("- user_state_embeddings.pkl")
print("================================================")