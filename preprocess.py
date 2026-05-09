"""
preprocess.py
=============
MovieLens 1M Preprocessing + SVD Embeddings

Outputs (saved to ./data/processed/):
    movie_embeddings.npy   - {movie_id: 32-dim SVD vector}
    user_profiles.npy      - {user_id: 32-dim weighted avg vector}
    user_ratings.npy       - {user_id: {movie_id: rating}}
    catalog.npy            - list of top-500 movie_ids
    movie_titles.npy       - {movie_id: title string} (for inspection)

Usage:
    python preprocess.py --data_dir ./data --top_k 500 --svd_dim 32
"""

import os
import argparse
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

# ─────────────────────────────────────────
# 1. Argument parsing
# ─────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess MovieLens 1M")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Directory containing ratings.dat and movies.dat")
    parser.add_argument("--top_k", type=int, default=500,
                        help="Number of top movies to keep in action catalog")
    parser.add_argument("--svd_dim", type=int, default=32,
                        help="Dimensionality of SVD embeddings")
    parser.add_argument("--min_user_ratings", type=int, default=20,
                        help="Minimum ratings a user must have to be included")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ─────────────────────────────────────────
# 2. Load raw data
# ─────────────────────────────────────────

def load_ratings(data_dir):
    """Load ratings.dat → DataFrame with columns [user_id, movie_id, rating]"""
    path = os.path.join(data_dir, "ratings.dat")
    ratings = pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
        encoding="latin-1"
    )
    ratings = ratings.drop(columns=["timestamp"])
    print(f"  Loaded {len(ratings):,} ratings | "
          f"{ratings.user_id.nunique():,} users | "
          f"{ratings.movie_id.nunique():,} movies")
    return ratings


def load_movies(data_dir):
    """Load movies.dat → DataFrame with columns [movie_id, title, genres]"""
    path = os.path.join(data_dir, "movies.dat")
    movies = pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["movie_id", "title", "genres"],
        encoding="latin-1"
    )
    print(f"  Loaded {len(movies):,} movies")
    return movies


# ─────────────────────────────────────────
# 3. Filter
# ─────────────────────────────────────────

def filter_top_movies(ratings, top_k):
    """Keep only the top_k most-rated movies."""
    movie_counts = ratings.groupby("movie_id")["rating"].count()
    top_movie_ids = movie_counts.nlargest(top_k).index.tolist()
    filtered = ratings[ratings.movie_id.isin(top_movie_ids)].copy()
    print(f"  Filtered to top {top_k} movies | {len(filtered):,} ratings remain")
    return filtered, top_movie_ids


def filter_active_users(ratings, min_ratings):
    """Keep only users who have rated at least min_ratings movies (within filtered set)."""
    user_counts = ratings.groupby("user_id")["rating"].count()
    active_users = user_counts[user_counts >= min_ratings].index
    filtered = ratings[ratings.user_id.isin(active_users)].copy()
    print(f"  Filtered to users with >= {min_ratings} ratings | "
          f"{filtered.user_id.nunique():,} users remain")
    return filtered


# ─────────────────────────────────────────
# 4. Build sparse rating matrix
# ─────────────────────────────────────────

def build_rating_matrix(ratings, top_movie_ids):
    """
    Build a sparse user x movie rating matrix.
    Rows = users, Columns = movies (in catalog order)
    Returns: sparse matrix, user_index list, movie_index list
    """
    # Create contiguous integer indices
    user_ids = sorted(ratings.user_id.unique())
    user_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    movie_to_idx = {mid: i for i, mid in enumerate(top_movie_ids)}

    # Only keep ratings for movies in catalog
    ratings_filtered = ratings[ratings.movie_id.isin(movie_to_idx)]

    row = ratings_filtered.user_id.map(user_to_idx).values
    col = ratings_filtered.movie_id.map(movie_to_idx).values
    data = ratings_filtered.rating.values.astype(np.float32)

    matrix = csr_matrix(
        (data, (row, col)),
        shape=(len(user_ids), len(top_movie_ids))
    )
    print(f"  Rating matrix shape: {matrix.shape} | "
          f"Sparsity: {100*(1 - matrix.nnz/np.prod(matrix.shape)):.1f}%")
    return matrix, user_ids, top_movie_ids


# ─────────────────────────────────────────
# 5. SVD embeddings
# ─────────────────────────────────────────

def run_svd(matrix, svd_dim, seed):
    """
    Run TruncatedSVD on the rating matrix.
    Returns:
        user_factors  : (n_users  x svd_dim)
        movie_factors : (n_movies x svd_dim)
    """
    print(f"  Running TruncatedSVD (n_components={svd_dim}) ...")
    svd = TruncatedSVD(n_components=svd_dim, random_state=seed)
    user_factors = svd.fit_transform(matrix)          # n_users  × svd_dim
    movie_factors = svd.components_.T                 # n_movies × svd_dim

    explained = svd.explained_variance_ratio_.sum()
    print(f"  SVD explained variance: {explained*100:.1f}%")

    # L2-normalize so cosine similarity = dot product (convenient for reward)
    user_factors  = normalize(user_factors,  norm="l2")
    movie_factors = normalize(movie_factors, norm="l2")

    return user_factors, movie_factors


# ─────────────────────────────────────────
# 6. Build output dictionaries
# ─────────────────────────────────────────

def build_movie_embeddings(movie_factors, top_movie_ids):
    """movie_id → 32-dim numpy vector"""
    return {mid: movie_factors[i] for i, mid in enumerate(top_movie_ids)}


def build_user_profiles(ratings, movie_embeddings, user_ids):
    """
    user_id → 32-dim preference vector
    = weighted average of SVD embeddings of rated movies, weighted by rating.
    Only movies in the catalog (movie_embeddings) are used.
    """
    profiles = {}
    catalog_set = set(movie_embeddings.keys())

    for uid in user_ids:
        user_df = ratings[(ratings.user_id == uid) &
                          (ratings.movie_id.isin(catalog_set))]
        if len(user_df) == 0:
            profiles[uid] = np.zeros(next(iter(movie_embeddings.values())).shape)
            continue

        vecs    = np.array([movie_embeddings[mid] for mid in user_df.movie_id])
        weights = user_df.rating.values.astype(np.float32)
        profile = np.average(vecs, axis=0, weights=weights)

        # L2 normalize
        norm = np.linalg.norm(profile)
        profiles[uid] = profile / norm if norm > 0 else profile

    return profiles


def build_user_ratings(ratings, top_movie_ids):
    """
    user_id → {movie_id: rating (float, 1-5)}
    Only includes movies in the catalog.
    """
    catalog_set = set(top_movie_ids)
    filtered = ratings[ratings.movie_id.isin(catalog_set)]
    user_ratings = (
        filtered.groupby("user_id")
        .apply(lambda df: dict(zip(df.movie_id, df.rating)))
        .to_dict()
    )
    return user_ratings


# ─────────────────────────────────────────
# 7. Save outputs
# ─────────────────────────────────────────

def save_outputs(out_dir, movie_embeddings, user_profiles,
                 user_ratings, catalog, movie_titles):
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "movie_embeddings.npy"), movie_embeddings)
    np.save(os.path.join(out_dir, "user_profiles.npy"),    user_profiles)
    np.save(os.path.join(out_dir, "user_ratings.npy"),     user_ratings)
    np.save(os.path.join(out_dir, "catalog.npy"),          catalog)
    np.save(os.path.join(out_dir, "movie_titles.npy"),     movie_titles)

    print(f"\n  Saved to {out_dir}/")
    print(f"    movie_embeddings : {len(movie_embeddings)} movies × "
          f"{next(iter(movie_embeddings.values())).shape[0]} dims")
    print(f"    user_profiles    : {len(user_profiles)} users")
    print(f"    user_ratings     : {len(user_ratings)} users")
    print(f"    catalog          : {len(catalog)} movies")


# ─────────────────────────────────────────
# 8. Quick sanity check
# ─────────────────────────────────────────

def sanity_check(movie_embeddings, user_profiles, catalog, movie_titles):
    """Print a few examples so you can verify things look right."""
    print("\n── Sanity Check ──────────────────────────────")

    # Show first 5 catalog movies
    print("First 5 catalog movies:")
    for mid in catalog[:5]:
        title = movie_titles.get(mid, "Unknown")
        vec   = movie_embeddings[mid]
        print(f"  [{mid}] {title:40s}  emb[:4]={np.round(vec[:4], 3)}")

    # Show nearest neighbors for catalog[0]
    anchor_id  = catalog[0]
    anchor_vec = movie_embeddings[anchor_id]
    sims = {mid: float(np.dot(anchor_vec, movie_embeddings[mid]))
            for mid in catalog}
    top5 = sorted(sims, key=sims.get, reverse=True)[:6]

    print(f"\nNearest neighbors to '{movie_titles.get(anchor_id)}':")
    for mid in top5:
        if mid == anchor_id:
            continue
        print(f"  {movie_titles.get(mid, mid):40s}  sim={sims[mid]:.3f}")

    # Sample user profile
    sample_uid = next(iter(user_profiles))
    print(f"\nSample user {sample_uid} profile[:4]: "
          f"{np.round(user_profiles[sample_uid][:4], 3)}")
    print("──────────────────────────────────────────────")


# ─────────────────────────────────────────
# 9. Main
# ─────────────────────────────────────────

def main():
    args = parse_args()
    np.random.seed(args.seed)

    out_dir = os.path.join(args.data_dir, "processed")

    print("\n=== Step 1: Loading raw data ===")
    ratings = load_ratings(args.data_dir)
    movies  = load_movies(args.data_dir)

    # Build title lookup (used for sanity check and evaluation)
    movie_titles = dict(zip(movies.movie_id, movies.title))

    print("\n=== Step 2: Filtering ===")
    ratings, top_movie_ids = filter_top_movies(ratings, args.top_k)
    ratings = filter_active_users(ratings, args.min_user_ratings)

    print("\n=== Step 3: Building rating matrix ===")
    matrix, user_ids, top_movie_ids = build_rating_matrix(ratings, top_movie_ids)

    print("\n=== Step 4: SVD embeddings ===")
    user_factors, movie_factors = run_svd(matrix, args.svd_dim, args.seed)

    print("\n=== Step 5: Building output structures ===")
    movie_embeddings = build_movie_embeddings(movie_factors, top_movie_ids)
    user_profiles    = build_user_profiles(ratings, movie_embeddings, user_ids)
    user_ratings     = build_user_ratings(ratings, top_movie_ids)
    catalog          = top_movie_ids   # list of 500 movie_ids, index = action

    print("\n=== Step 6: Saving ===")
    save_outputs(out_dir, movie_embeddings, user_profiles,
                 user_ratings, catalog, movie_titles)

    sanity_check(movie_embeddings, user_profiles, catalog, movie_titles)

    print("\n✓ Preprocessing complete.\n")


if __name__ == "__main__":
    main()
