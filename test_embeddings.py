import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

movie_embeddings = np.load("data/processed/movie_embeddings.npy", allow_pickle=True).item()
movie_titles     = np.load("data/processed/movie_titles.npy",     allow_pickle=True).item()
catalog          = np.load("data/processed/catalog.npy",           allow_pickle=True).tolist()
user_profiles    = np.load("data/processed/user_profiles.npy",    allow_pickle=True).item()
user_ratings     = np.load("data/processed/user_ratings.npy",     allow_pickle=True).item()

emb_matrix = np.array([movie_embeddings[mid] for mid in catalog])

# ── 1. Neighbor check ─────────────────────────────────────────────────────
def nearest(movie_id, n=5):
    anchor = movie_embeddings[movie_id]
    sims = {m: float(np.dot(anchor, movie_embeddings[m])) for m in catalog}
    top = sorted(sims, key=sims.get, reverse=True)[1:n+1]
    for mid in top:
        print(f"    {movie_titles[mid]:45s}  {sims[mid]:.3f}")

print("=== 1. Neighbor Check ===")
for name, mid in [
    ("Schindler's List",  527),
    ("The Matrix",       2571),
    ("Forrest Gump",      356),
    ("Toy Story",           1),
    ("Pulp Fiction",      296),
    ("Star Wars IV",      260),
]:
    if mid not in movie_embeddings:
        print(f"  {name} not in catalog")
        continue
    print(f"\n  {name} neighbors:")
    nearest(mid)

# ── 2. Embedding spread ───────────────────────────────────────────────────
all_sims = cosine_similarity(emb_matrix)
np.fill_diagonal(all_sims, 0)
print(f"\n=== 2. Embedding Spread (500 movies) ===")
print(f"  Mean pairwise similarity : {all_sims.mean():.4f}  (want < 0.3)")
print(f"  Max pairwise similarity  : {all_sims.max():.4f}")
print(f"  Min pairwise similarity  : {all_sims.min():.4f}")
print(f"  Std pairwise similarity  : {all_sims.std():.4f}  (want > 0.1, means spread)")

# ── 3. User coverage ──────────────────────────────────────────────────────
rated_counts = [len(r) for r in user_ratings.values()]
print(f"\n=== 3. User Coverage (all {len(user_ratings)} users) ===")
print(f"  Avg movies rated per user : {np.mean(rated_counts):.1f}")
print(f"  Median                    : {np.median(rated_counts):.1f}")
print(f"  Min                       : {np.min(rated_counts)}")
print(f"  Max                       : {np.max(rated_counts)}")
print(f"  Users with < 5 ratings    : {sum(1 for c in rated_counts if c < 5)}")
print(f"  Users with >= 20 ratings  : {sum(1 for c in rated_counts if c >= 20)}")

# ── 4. Reward signal quality (ALL users) ─────────────────────────────────
print(f"\n=== 4. Reward Signal Quality (all users) ===")
correlations = []
zero_variance_users = 0
insufficient_users  = 0

for uid, rated in user_ratings.items():
    if len(rated) < 5:
        insufficient_users += 1
        continue
    profile = user_profiles[uid]
    sims = [float(np.dot(profile, movie_embeddings[mid])) for mid in rated]
    rats = [rated[mid] / 5.0 for mid in rated]
    if np.std(sims) == 0 or np.std(rats) == 0:
        zero_variance_users += 1
        continue
    corr = np.corrcoef(sims, rats)[0, 1]
    correlations.append(corr)

correlations = np.array(correlations)
print(f"  Users evaluated           : {len(correlations)}")
print(f"  Skipped (< 5 ratings)     : {insufficient_users}")
print(f"  Skipped (zero variance)   : {zero_variance_users}")
print(f"  Mean correlation          : {correlations.mean():.4f}  (> 0.1 = good)")
print(f"  Median correlation        : {np.median(correlations):.4f}")
print(f"  Std correlation           : {correlations.std():.4f}")
print(f"  % users with corr > 0.1  : {100*np.mean(correlations > 0.1):.1f}%")
print(f"  % users with corr > 0.2  : {100*np.mean(correlations > 0.2):.1f}%")
print(f"  % users with corr < 0    : {100*np.mean(correlations < 0):.1f}%  (profile misleading for these users)")

# ── 5. Profile match vs actual ratings (ALL users, top-5 accuracy) ────────
print(f"\n=== 5. Profile Top-5 Accuracy (all users) ===")
print("  (Does the user's highest-rated movie appear in their top-20 profile matches?)")
hit_at_5  = []
hit_at_10 = []
hit_at_20 = []

for uid, rated in user_ratings.items():
    if len(rated) < 5:
        continue
    profile = user_profiles[uid]
    # Their actual top-rated movie
    top_actual = sorted(rated, key=rated.get, reverse=True)[0]

    # Profile's top matches
    sims = {m: float(np.dot(profile, movie_embeddings[m])) for m in catalog}
    top_by_profile = sorted(sims, key=sims.get, reverse=True)

    hit_at_5.append(top_actual  in top_by_profile[:5])
    hit_at_10.append(top_actual in top_by_profile[:10])
    hit_at_20.append(top_actual in top_by_profile[:20])

print(f"  Hit@5  : {100*np.mean(hit_at_5):.1f}%  (top rated movie in top-5 profile matches)")
print(f"  Hit@10 : {100*np.mean(hit_at_10):.1f}%")
print(f"  Hit@20 : {100*np.mean(hit_at_20):.1f}%")

# ── 6. Diversity of user profiles ────────────────────────────────────────
print(f"\n=== 6. User Profile Diversity ===")
profile_matrix = np.array(list(user_profiles.values()))
profile_sims   = cosine_similarity(profile_matrix)
np.fill_diagonal(profile_sims, 0)
print(f"  Mean pairwise user profile similarity : {profile_sims.mean():.4f}")
print(f"  (Low = users are diverse from each other = good for training)")

print("\n✓ Full embedding check complete.")