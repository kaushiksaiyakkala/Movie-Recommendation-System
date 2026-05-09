"""
preprocess.py
=============
MovieLens 1M Preprocessing + Item2Vec Embeddings

Item2Vec treats each user's rating history as a sentence and each movie as a word.
Movies that appear in similar users' histories end up close in embedding space.
This captures co-occurrence patterns more relevant to sequential recommendation
than SVD's rating reconstruction objective.

Reference: Barkan & Koenigstein, "Item2Vec: Neural Item Embedding for CF", 2016.

Outputs (saved to ./data/processed/):
    movie_embeddings.npy   - {movie_id: 32-dim Item2Vec vector}
    user_profiles.npy      - {user_id: 32-dim weighted avg vector}
    user_ratings.npy       - {user_id: {movie_id: rating (1-5)}}
    catalog.npy            - list of top-500 movie_ids (index = action)
    movie_titles.npy       - {movie_id: title string}

Usage:
    python preprocess.py --data_dir ./data --top_k 500 --embed_dim 32
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from gensim.models import Word2Vec

# ─────────────────────────────────────────
# 1. Argument parsing
# ─────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess MovieLens 1M with Item2Vec")
    parser.add_argument("--data_dir",          type=str, default="./data")
    parser.add_argument("--top_k",             type=int, default=500,
                        help="Size of movie action catalog")
    parser.add_argument("--embed_dim",         type=int, default=32,
                        help="Embedding dimensionality")
    parser.add_argument("--window",            type=int, default=5,
                        help="Item2Vec context window size")
    parser.add_argument("--min_user_ratings",  type=int, default=20,
                        help="Min ratings a user must have to be included")
    parser.add_argument("--epochs",            type=int, default=20,
                        help="Item2Vec training epochs")
    parser.add_argument("--seed",              type=int, default=42)
    return parser.parse_args()


# ─────────────────────────────────────────
# 2. Load raw data
# ─────────────────────────────────────────

def load_ratings(data_dir):
    path = os.path.join(data_dir, "ratings.dat")
    ratings = pd.read_csv(
        path, sep="::", engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
        encoding="latin-1"
    )
    ratings = ratings.drop(columns=["timestamp"])
    print(f"  Loaded {len(ratings):,} ratings | "
          f"{ratings.user_id.nunique():,} users | "
          f"{ratings.movie_id.nunique():,} movies")
    return ratings


def load_movies(data_dir):
    path = os.path.join(data_dir, "movies.dat")
    movies = pd.read_csv(
        path, sep="::", engine="python",
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
    """Keep only users with at least min_ratings within the filtered catalog."""
    user_counts = ratings.groupby("user_id")["rating"].count()
    active_users = user_counts[user_counts >= min_ratings].index
    filtered = ratings[ratings.user_id.isin(active_users)].copy()
    print(f"  Filtered to users with >= {min_ratings} ratings | "
          f"{filtered.user_id.nunique():,} users remain")
    return filtered


# ─────────────────────────────────────────
# 4. Build Item2Vec sequences
# ─────────────────────────────────────────

def build_sequences(ratings):
    """
    For each user, build a sequence of movie_ids ordered by rating (high to low).
    High-rated movies appear together — the model learns taste co-occurrence.
    Each sequence is a list of strings (Word2Vec expects string tokens).
    """
    sequences = []
    grouped = ratings.groupby("user_id")

    for uid, group in grouped:
        sorted_movies = group.sort_values(
            ["rating", "movie_id"], ascending=[False, True]
        )["movie_id"].tolist()
        sequences.append([str(mid) for mid in sorted_movies])

    print(f"  Built {len(sequences):,} user sequences")
    print(f"  Average sequence length: {np.mean([len(s) for s in sequences]):.1f} movies")
    return sequences


# ─────────────────────────────────────────
# 5. Train Item2Vec
# ─────────────────────────────────────────

def train_item2vec(sequences, embed_dim, window, epochs, seed):
    """
    Train Word2Vec (skip-gram) on user sequences = Item2Vec.
    sg=1      → skip-gram (better for rare items)
    negative=5 → negative sampling
    """
    print(f"  Training Item2Vec (dim={embed_dim}, window={window}, epochs={epochs}) ...")
    model = Word2Vec(
        sentences=sequences,
        vector_size=embed_dim,
        window=window,
        min_count=1,
        workers=4,
        sg=1,
        hs=0,
        negative=5,
        epochs=epochs,
        seed=seed
    )
    print(f"  Item2Vec vocab size: {len(model.wv):,} movies")
    return model


# ─────────────────────────────────────────
# 6. Build output dictionaries
# ─────────────────────────────────────────

def build_movie_embeddings(model, top_movie_ids, embed_dim):
    """
    movie_id (int) → 32-dim L2-normalized numpy vector.
    Missing vocab entries get a zero vector (fallback, rare).
    """
    embeddings = {}
    missing = 0
    for mid in top_movie_ids:
        key = str(mid)
        if key in model.wv:
            embeddings[mid] = model.wv[key].astype(np.float32)
        else:
            embeddings[mid] = np.zeros(embed_dim, dtype=np.float32)
            missing += 1

    if missing:
        print(f"  Warning: {missing} movies missing from vocab (zero vector used)")

    # L2 normalize — dot product now equals cosine similarity
    vecs = np.array([embeddings[mid] for mid in top_movie_ids])
    vecs = normalize(vecs, norm="l2")
    for i, mid in enumerate(top_movie_ids):
        embeddings[mid] = vecs[i]

    return embeddings


def build_user_profiles(ratings, movie_embeddings, embed_dim):
    """
    user_id → 32-dim preference vector.
    Weighted average of Item2Vec embeddings of movies the user rated,
    weighted by their rating. L2 normalized.
    """
    profiles = {}
    catalog_set = set(movie_embeddings.keys())
    grouped = ratings[ratings.movie_id.isin(catalog_set)].groupby("user_id")

    for uid, group in grouped:
        vecs    = np.array([movie_embeddings[mid] for mid in group.movie_id])
        weights = group.rating.values.astype(np.float32)
        profile = np.average(vecs, axis=0, weights=weights)
        norm    = np.linalg.norm(profile)
        profiles[uid] = (profile / norm).astype(np.float32) if norm > 0 else profile.astype(np.float32)

    print(f"  Built profiles for {len(profiles):,} users")
    return profiles


def build_user_ratings(ratings, top_movie_ids):
    """
    user_id → {movie_id: rating (float, 1-5)}
    Only catalog movies included.
    """
    catalog_set = set(top_movie_ids)
    filtered = ratings[ratings.movie_id.isin(catalog_set)]
    user_ratings = (
        filtered.groupby("user_id")
        .apply(lambda df: dict(zip(df.movie_id, df.rating.astype(float))))
        .to_dict()
    )
    return user_ratings


# ─────────────────────────────────────────
# 7. Save
# ─────────────────────────────────────────

def save_outputs(out_dir, movie_embeddings, user_profiles,
                 user_ratings, catalog, movie_titles):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "movie_embeddings.npy"), movie_embeddings)
    np.save(os.path.join(out_dir, "user_profiles.npy"),    user_profiles)
    np.save(os.path.join(out_dir, "user_ratings.npy"),     user_ratings)
    np.save(os.path.join(out_dir, "catalog.npy"),          catalog)
    np.save(os.path.join(out_dir, "movie_titles.npy"),     movie_titles)

    sample_vec = next(iter(movie_embeddings.values()))
    print(f"\n  Saved to {out_dir}/")
    print(f"    movie_embeddings : {len(movie_embeddings)} movies × {sample_vec.shape[0]} dims")
    print(f"    user_profiles    : {len(user_profiles)} users")
    print(f"    user_ratings     : {len(user_ratings)} users")
    print(f"    catalog          : {len(catalog)} movies")


# ─────────────────────────────────────────
# 8. Sanity check
# ─────────────────────────────────────────

def sanity_check(movie_embeddings, user_profiles, catalog, movie_titles):
    print("\n── Sanity Check ───────────────────────────────────────────")

    # First 5 catalog movies
    print("First 5 catalog movies:")
    for mid in catalog[:5]:
        title = movie_titles.get(mid, "Unknown")
        vec   = movie_embeddings[mid]
        print(f"  [{mid:5d}] {title:45s}  emb[:4]={np.round(vec[:4], 3)}")

    # Nearest neighbors for well-known movies
    test_movies = {"Toy Story": 1, "Star Wars IV": 260, "Pulp Fiction": 296}

    for name, mid in test_movies.items():
        if mid not in movie_embeddings:
            print(f"\n  '{name}' (id={mid}) not in catalog, skipping.")
            continue
        anchor = movie_embeddings[mid]
        sims   = {m: float(np.dot(anchor, movie_embeddings[m])) for m in catalog}
        top5   = sorted(sims, key=sims.get, reverse=True)[1:6]

        print(f"\nNearest neighbors to '{name}' (id={mid}):")
        for nid in top5:
            print(f"  {movie_titles.get(nid, nid):45s}  sim={sims[nid]:.3f}")

    # Sample user
    sample_uid = next(iter(user_profiles))
    print(f"\nSample user {sample_uid} profile[:4]: "
          f"{np.round(user_profiles[sample_uid][:4], 3)}")
    print("───────────────────────────────────────────────────────────")


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
    movie_titles = dict(zip(movies.movie_id, movies.title))

    print("\n=== Step 2: Filtering ===")
    ratings, top_movie_ids = filter_top_movies(ratings, args.top_k)
    ratings = filter_active_users(ratings, args.min_user_ratings)

    print("\n=== Step 3: Building Item2Vec sequences ===")
    sequences = build_sequences(ratings)

    print("\n=== Step 4: Training Item2Vec ===")
    model = train_item2vec(sequences, args.embed_dim, args.window, args.epochs, args.seed)

    print("\n=== Step 5: Building output structures ===")
    movie_embeddings = build_movie_embeddings(model, top_movie_ids, args.embed_dim)
    user_profiles    = build_user_profiles(ratings, movie_embeddings, args.embed_dim)
    user_ratings     = build_user_ratings(ratings, top_movie_ids)
    catalog          = top_movie_ids

    print("\n=== Step 6: Saving ===")
    save_outputs(out_dir, movie_embeddings, user_profiles,
                 user_ratings, catalog, movie_titles)

    sanity_check(movie_embeddings, user_profiles, catalog, movie_titles)

    print("\n✓ Preprocessing complete.\n")


if __name__ == "__main__":
    main()
