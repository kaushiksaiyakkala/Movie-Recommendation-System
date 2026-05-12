# ============================================================
# MOVIELENS 32M PREPROCESSING PIPELINE
# ============================================================
# GOAL:
# 1. Load ratings + movies
# 2. Keep top 5000 most interacted movies
# 3. Keep users with >= 20 interactions
# 4. Build chronological user sequences
# 5. Encode movie IDs
# 6. Save processed datasets
#
# OUTPUT FILES:
# - processed_ratings.csv
# - movie_id_map.pkl
# - user_sequences.pkl
# ============================================================

import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import pickle
from sklearn.preprocessing import LabelEncoder

# ============================================================
# PATHS
# ============================================================

RATINGS_PATH = "C:\\Users\\deepa\\OneDrive\\Desktop\\RL project\\ml-32m\\ratings.csv"
MOVIES_PATH = "C:\\Users\\deepa\\OneDrive\\Desktop\\RL project\\ml-32m\\movies.csv"

# ============================================================
# LOAD DATA
# ============================================================

print("Loading data...")

ratings = pd.read_csv(RATINGS_PATH)
movies = pd.read_csv(MOVIES_PATH)

print("Ratings shape:", ratings.shape)
print("Movies shape:", movies.shape)

# Expected columns:
# ratings: userId, movieId, rating, timestamp
# movies : movieId, title, genres

# ============================================================
# KEEP TOP 5000 MOVIES
# ============================================================

TOP_K_MOVIES = 5000

print(f"\nSelecting top {TOP_K_MOVIES} most interacted movies...")

movie_counts = ratings["movieId"].value_counts()

top_movies = movie_counts.head(TOP_K_MOVIES).index

ratings = ratings[ratings["movieId"].isin(top_movies)]

print("Remaining ratings:", len(ratings))

# ============================================================
# REMOVE LOW-ACTIVITY USERS
# ============================================================

MIN_INTERACTIONS = 20

print(f"\nFiltering users with < {MIN_INTERACTIONS} interactions...")

user_counts = ratings["userId"].value_counts()

valid_users = user_counts[user_counts >= MIN_INTERACTIONS].index

ratings = ratings[ratings["userId"].isin(valid_users)]

print("Remaining users:", ratings["userId"].nunique())
print("Remaining ratings:", len(ratings))

# ============================================================
# SORT CHRONOLOGICALLY
# ============================================================

print("\nSorting interactions chronologically...")

ratings = ratings.sort_values(
    by=["userId", "timestamp"]
).reset_index(drop=True)

# ============================================================
# ENCODE MOVIE IDS
# ============================================================

print("\nEncoding movie IDs...")

movie_encoder = LabelEncoder()

ratings["movie_idx"] = movie_encoder.fit_transform(
    ratings["movieId"]
)

num_movies = ratings["movie_idx"].nunique()

print("Encoded movies:", num_movies)

# ============================================================
# ENCODE USER IDS
# ============================================================

print("\nEncoding user IDs...")

user_encoder = LabelEncoder()

ratings["user_idx"] = user_encoder.fit_transform(
    ratings["userId"]
)

num_users = ratings["user_idx"].nunique()

print("Encoded users:", num_users)

# ============================================================
# MERGE MOVIE METADATA
# ============================================================

print("\nMerging movie metadata...")

movies = movies[["movieId", "title", "genres"]]

ratings = ratings.merge(
    movies,
    on="movieId",
    how="left"
)

# ============================================================
# CREATE BINARY TARGET
# ============================================================
# Positive interaction:
# rating >= 4

ratings["liked"] = (ratings["rating"] >= 4).astype(int)

# ============================================================
# BUILD USER SEQUENCES
# ============================================================

print("\nBuilding user sequences...")

user_sequences = defaultdict(list)

for row in tqdm(ratings.itertuples(), total=len(ratings)):

    user_sequences[row.user_idx].append({
        "movie_idx": int(row.movie_idx),
        "rating": float(row.rating),
        "liked": int(row.liked),
        "timestamp": int(row.timestamp)
    })

print("Built sequences for", len(user_sequences), "users")

# ============================================================
# SAVE PROCESSED RATINGS
# ============================================================

print("\nSaving processed_ratings.csv...")

ratings.to_csv(
    "processed_ratings.csv",
    index=False
)

# ============================================================
# SAVE MOVIE ENCODER
# ============================================================

print("\nSaving movie_id_map.pkl...")

with open("movie_id_map.pkl", "wb") as f:
    pickle.dump(movie_encoder, f)

# ============================================================
# SAVE USER ENCODER
# ============================================================

print("\nSaving user_id_map.pkl...")

with open("user_id_map.pkl", "wb") as f:
    pickle.dump(user_encoder, f)

# ============================================================
# SAVE USER SEQUENCES
# ============================================================

print("\nSaving user_sequences.pkl...")

with open("user_sequences.pkl", "wb") as f:
    pickle.dump(dict(user_sequences), f)

# ============================================================
# FINAL STATS
# ============================================================

print("\n================ FINAL DATASET STATS ================")

print("Users      :", num_users)
print("Movies     :", num_movies)
print("Interactions:", len(ratings))

avg_len = np.mean([len(v) for v in user_sequences.values()])

print("Avg sequence length:", round(avg_len, 2))

print("\nPreprocessing complete.")