# ============================================================
# STAGE 5 — CANDIDATE GENERATION WITH FAISS
# ============================================================
#
# GOAL:
# Build fast retrieval system:
#
# user_state -> top candidate movies
#
# RL will NOT search all movies.
# RL will only rerank top-K candidates.
#
# INPUT:
# - movie_embeddings.npy
#
# OUTPUT:
# - faiss_movie_index.bin
#
# ============================================================

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import faiss
import numpy as np
import pickle

# ============================================================
# CONFIG
# ============================================================

TOP_K = 100

# ============================================================
# LOAD MOVIE EMBEDDINGS
# ============================================================

print("\nLoading movie embeddings...")

movie_embeddings = np.load(
    "C:\\Users\\deepa\\OneDrive\\Desktop\\RL project\\movie_embeddings.npy"
).astype("float32")

print("Movie embedding shape:", movie_embeddings.shape)

NUM_MOVIES = movie_embeddings.shape[0]
EMBED_DIM = movie_embeddings.shape[1]

# ============================================================
# NORMALIZE EMBEDDINGS
# ============================================================
# IMPORTANT:
# Enables cosine similarity search
# using inner product index

print("\nNormalizing embeddings...")

faiss.normalize_L2(movie_embeddings)

# ============================================================
# BUILD FAISS INDEX
# ============================================================

print("\nBuilding FAISS index...")

index = faiss.IndexFlatIP(
    EMBED_DIM
)

index.add(movie_embeddings)

print("Movies indexed:", index.ntotal)

# ============================================================
# SAVE INDEX
# ============================================================

print("\nSaving FAISS index...")

faiss.write_index(
    index,
    "C:\\Users\\deepa\\OneDrive\\Desktop\\RL project\\faiss_movie_index.bin"
)

# ============================================================
# TEST RETRIEVAL
# ============================================================

print("\nTesting retrieval...")

test_movie = movie_embeddings[0].reshape(1, -1)

scores, indices = index.search(
    test_movie,
    TOP_K
)

print("\nTop retrieved movie indices:")

print(indices[0][:10])

print("\nSimilarity scores:")

print(scores[0][:10])

# ============================================================
# OPTIONAL:
# SAVE NORMALIZED EMBEDDINGS
# ============================================================

np.save(
    "C:\\Users\\deepa\\OneDrive\\Desktop\\RL project\\movie_embeddings_normalized.npy",
    movie_embeddings
)

# ============================================================
# DONE
# ============================================================

print("\n================================================")
print("FAISS INDEX COMPLETE")
print("Saved:")
print("- faiss_movie_index.bin")
print("- movie_embeddings_normalized.npy")
print("================================================")