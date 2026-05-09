"""
preprocess.py
=============
MovieLens 1M Preprocessing + BPR (Bayesian Personalized Ranking) Embeddings

BPR directly optimizes for ranking: it learns that a user prefers movie A
over movie B by training on (user, positive_movie, negative_movie) triples.
This is more appropriate for recommendation than Item2Vec (co-occurrence)
or SVD (rating reconstruction) because we care about relative preference,
not absolute rating prediction.

Reference: Rendle et al., "BPR: Bayesian Personalized Ranking from Implicit
Feedback", UAI 2009.

Outputs (saved to ./data/processed/):
    movie_embeddings.npy   - {movie_id: 64-dim BPR vector}
    user_profiles.npy      - {user_id: 64-dim BPR user vector}
    user_ratings.npy       - {user_id: {movie_id: rating (1-5)}}
    catalog.npy            - list of top-500 movie_ids (index = action)
    movie_titles.npy       - {movie_id: title string}

Usage:
    python preprocess.py --data_dir ./data --top_k 500 --embed_dim 64
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import normalize

# ─────────────────────────────────────────
# 1. Args
# ─────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess MovieLens 1M with BPR")
    parser.add_argument("--data_dir",         type=str,   default="./data")
    parser.add_argument("--top_k",            type=int,   default=500)
    parser.add_argument("--embed_dim",        type=int,   default=64)
    parser.add_argument("--pos_threshold",    type=float, default=4.0,
                        help="Ratings >= this are positive samples")
    parser.add_argument("--neg_threshold",    type=float, default=3.0,
                        help="Ratings < this are negative samples")
    parser.add_argument("--epochs",           type=int,   default=50)
    parser.add_argument("--batch_size",       type=int,   default=2048)
    parser.add_argument("--lr",               type=float, default=0.005)
    parser.add_argument("--reg",              type=float, default=0.001,
                        help="L2 regularization weight")
    parser.add_argument("--min_user_ratings", type=int,   default=5,
                        help="Min positive ratings a user must have")
    parser.add_argument("--seed",             type=int,   default=42)
    return parser.parse_args()


# ─────────────────────────────────────────
# 2. Load data
# ─────────────────────────────────────────

def load_ratings(data_dir):
    path = os.path.join(data_dir, "ratings.dat")
    ratings = pd.read_csv(
        path, sep="::", engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
        encoding="latin-1"
    ).drop(columns=["timestamp"])
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
    counts        = ratings.groupby("movie_id")["rating"].count()
    top_movie_ids = counts.nlargest(top_k).index.tolist()
    filtered      = ratings[ratings.movie_id.isin(top_movie_ids)].copy()
    print(f"  Filtered to top {top_k} movies | {len(filtered):,} ratings remain")
    return filtered, top_movie_ids


def filter_active_users(ratings, pos_threshold, min_ratings):
    """Keep users who have at least min_ratings POSITIVE ratings in catalog."""
    pos     = ratings[ratings.rating >= pos_threshold]
    counts  = pos.groupby("user_id")["rating"].count()
    keep    = counts[counts >= min_ratings].index
    filtered = ratings[ratings.user_id.isin(keep)].copy()
    print(f"  Filtered to users with >= {min_ratings} positive ratings | "
          f"{filtered.user_id.nunique():,} users remain")
    return filtered


# ─────────────────────────────────────────
# 4. Build index mappings
# ─────────────────────────────────────────

def build_mappings(ratings, top_movie_ids):
    user_ids    = sorted(ratings.user_id.unique())
    user_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    movie_to_idx = {mid: i for i, mid in enumerate(top_movie_ids)}
    idx_to_movie = {i: mid for mid, i in movie_to_idx.items()}
    return user_ids, user_to_idx, movie_to_idx, idx_to_movie


# ─────────────────────────────────────────
# 5. BPR Dataset
# ─────────────────────────────────────────

class BPRDataset(Dataset):
    """
    Each sample is a triple: (user_idx, pos_movie_idx, neg_movie_idx)
    Positive = movies rated >= pos_threshold
    Negative = movies rated < neg_threshold OR unrated (sampled randomly)
    """
    def __init__(self, ratings, user_to_idx, movie_to_idx,
                 pos_threshold, neg_threshold, n_movies, seed=42):
        np.random.seed(seed)
        self.n_movies  = n_movies
        self.triples   = []

        catalog_set = set(movie_to_idx.keys())

        grouped = ratings[ratings.movie_id.isin(catalog_set)].groupby("user_id")

        for uid, group in grouped:
            uidx = user_to_idx[uid]

            pos_movies = group[group.rating >= pos_threshold]["movie_id"].tolist()
            neg_movies = group[group.rating <  neg_threshold]["movie_id"].tolist()

            # Convert to indices
            pos_idxs = [movie_to_idx[m] for m in pos_movies if m in movie_to_idx]
            neg_idxs = [movie_to_idx[m] for m in neg_movies if m in movie_to_idx]

            if len(pos_idxs) == 0:
                continue

            # For each positive, pair with a negative (rated or random unrated)
            all_movie_idxs = set(range(n_movies))
            unrated_idxs   = list(all_movie_idxs - set(pos_idxs) - set(neg_idxs))

            neg_pool = neg_idxs + unrated_idxs  # prefer rated negatives, pad with unrated

            for pidx in pos_idxs:
                nidx = neg_pool[np.random.randint(len(neg_pool))]
                self.triples.append((uidx, pidx, nidx))

        print(f"  BPR dataset: {len(self.triples):,} training triples")

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        u, p, n = self.triples[idx]
        return torch.tensor(u), torch.tensor(p), torch.tensor(n)


# ─────────────────────────────────────────
# 6. BPR Model
# ─────────────────────────────────────────

class BPRModel(nn.Module):
    def __init__(self, n_users, n_movies, embed_dim):
        super().__init__()
        self.user_embeddings  = nn.Embedding(n_users,  embed_dim)
        self.movie_embeddings = nn.Embedding(n_movies, embed_dim)

        # Initialize with small random values
        nn.init.normal_(self.user_embeddings.weight,  std=0.01)
        nn.init.normal_(self.movie_embeddings.weight, std=0.01)

    def forward(self, user_idxs, pos_idxs, neg_idxs):
        u   = self.user_embeddings(user_idxs)   # (B, D)
        p   = self.movie_embeddings(pos_idxs)   # (B, D)
        n   = self.movie_embeddings(neg_idxs)   # (B, D)

        pos_scores = (u * p).sum(dim=1)         # (B,)
        neg_scores = (u * n).sum(dim=1)         # (B,)

        return pos_scores, neg_scores

    def bpr_loss(self, pos_scores, neg_scores, reg, user_idxs, pos_idxs, neg_idxs):
        """
        BPR loss: -log(sigmoid(pos_score - neg_score)) + L2 regularization
        Maximizes the gap between positive and negative scores.
        """
        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()

        # L2 regularization on embeddings
        u_emb = self.user_embeddings(user_idxs)
        p_emb = self.movie_embeddings(pos_idxs)
        n_emb = self.movie_embeddings(neg_idxs)
        reg_loss = reg * (u_emb.norm(2).pow(2) +
                          p_emb.norm(2).pow(2) +
                          n_emb.norm(2).pow(2)) / user_idxs.size(0)

        return loss + reg_loss


# ─────────────────────────────────────────
# 7. Train BPR
# ─────────────────────────────────────────

def train_bpr(dataset, n_users, n_movies, embed_dim, epochs, batch_size, lr, reg, seed):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Training on: {device}")

    model     = BPRModel(n_users, n_movies, embed_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader    = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    print(f"  Training BPR (dim={embed_dim}, epochs={epochs}, "
          f"batch={batch_size}, lr={lr}) ...")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for user_idxs, pos_idxs, neg_idxs in loader:
            user_idxs = user_idxs.to(device)
            pos_idxs  = pos_idxs.to(device)
            neg_idxs  = neg_idxs.to(device)

            optimizer.zero_grad()
            pos_scores, neg_scores = model(user_idxs, pos_idxs, neg_idxs)
            loss = model.bpr_loss(pos_scores, neg_scores, reg,
                                  user_idxs, pos_idxs, neg_idxs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0 or epoch == 1:
            avg_loss = total_loss / len(loader)
            print(f"  Epoch {epoch:3d}/{epochs}  loss={avg_loss:.4f}")

    return model


# ─────────────────────────────────────────
# 8. Extract embeddings
# ─────────────────────────────────────────

def extract_embeddings(model, user_ids, user_to_idx, top_movie_ids, movie_to_idx):
    model.eval()
    with torch.no_grad():
        all_movie_embs = model.movie_embeddings.weight.cpu().numpy()  # (n_movies, D)
        all_user_embs  = model.user_embeddings.weight.cpu().numpy()   # (n_users,  D)

    # L2 normalize — dot product = cosine similarity
    all_movie_embs = normalize(all_movie_embs, norm="l2")
    all_user_embs  = normalize(all_user_embs,  norm="l2")

    # movie_id → embedding
    movie_embeddings = {
        mid: all_movie_embs[movie_to_idx[mid]]
        for mid in top_movie_ids
    }

    # user_id → embedding (this IS the BPR user profile)
    user_profiles = {
        uid: all_user_embs[user_to_idx[uid]]
        for uid in user_ids
    }

    return movie_embeddings, user_profiles


# ─────────────────────────────────────────
# 9. Save
# ─────────────────────────────────────────

def build_user_ratings(ratings, top_movie_ids):
    catalog_set = set(top_movie_ids)
    filtered    = ratings[ratings.movie_id.isin(catalog_set)]
    return (
        filtered.groupby("user_id")
        .apply(lambda df: dict(zip(df.movie_id, df.rating.astype(float))))
        .to_dict()
    )


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
# 10. Sanity check
# ─────────────────────────────────────────

def sanity_check(movie_embeddings, user_profiles, catalog, movie_titles):
    print("\n── Sanity Check ────────────────────────────────────────────")

    print("First 5 catalog movies:")
    for mid in catalog[:5]:
        vec = movie_embeddings[mid]
        print(f"  [{mid:5d}] {movie_titles.get(mid,'?'):45s}  emb[:4]={np.round(vec[:4],3)}")

    test_movies = {
        "Toy Story":      1,
        "Star Wars IV":   260,
        "Pulp Fiction":   296,
        "The Matrix":     2571,
        "Forrest Gump":   356,
    }
    for name, mid in test_movies.items():
        if mid not in movie_embeddings:
            print(f"\n  '{name}' not in catalog")
            continue
        anchor = movie_embeddings[mid]
        sims   = {m: float(np.dot(anchor, movie_embeddings[m])) for m in catalog}
        top5   = sorted(sims, key=sims.get, reverse=True)[1:6]
        print(f"\nNearest neighbors to '{name}':")
        for nid in top5:
            print(f"  {movie_titles.get(nid, nid):45s}  sim={sims[nid]:.3f}")

    sample_uid = next(iter(user_profiles))
    print(f"\nSample user {sample_uid} profile[:4]: "
          f"{np.round(user_profiles[sample_uid][:4], 3)}")
    print("────────────────────────────────────────────────────────────")


# ─────────────────────────────────────────
# 11. Main
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
    ratings = filter_active_users(ratings, args.pos_threshold, args.min_user_ratings)

    print("\n=== Step 3: Building index mappings ===")
    user_ids, user_to_idx, movie_to_idx, idx_to_movie = build_mappings(ratings, top_movie_ids)
    n_users  = len(user_ids)
    n_movies = len(top_movie_ids)
    print(f"  {n_users} users | {n_movies} movies")

    print("\n=== Step 4: Building BPR dataset ===")
    dataset = BPRDataset(
        ratings, user_to_idx, movie_to_idx,
        args.pos_threshold, args.neg_threshold,
        n_movies, args.seed
    )

    print("\n=== Step 5: Training BPR ===")
    model = train_bpr(
        dataset, n_users, n_movies, args.embed_dim,
        args.epochs, args.batch_size, args.lr, args.reg, args.seed
    )

    print("\n=== Step 6: Extracting embeddings ===")
    movie_embeddings, user_profiles = extract_embeddings(
        model, user_ids, user_to_idx, top_movie_ids, movie_to_idx
    )

    print("\n=== Step 7: Building user ratings dict ===")
    user_ratings = build_user_ratings(ratings, top_movie_ids)

    print("\n=== Step 8: Saving ===")
    save_outputs(out_dir, movie_embeddings, user_profiles,
                 user_ratings, top_movie_ids, movie_titles)

    sanity_check(movie_embeddings, user_profiles, top_movie_ids, movie_titles)

    print("\n✓ Preprocessing complete.\n")


if __name__ == "__main__":
    main()
