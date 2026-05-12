"""
evaluate.py
===========
Comprehensive evaluation with proper diversity metrics.

Diversity metrics computed:
  1. ILD (Intra-List Diversity) — mean pairwise cosine distance between
     all recommended movies in an episode. Standard metric in RecSys
     literature. Range [0,1], higher = more diverse.
     Reference: Ziegler et al., "Improving Recommendation Lists Through
     Topic Diversification", WWW 2005.

  2. Unique genres per episode — how many distinct genres covered.

  3. Mean embedding diversity bonus — per-step diversity from environment.

  4. Coverage — fraction of catalog recommended across all episodes.

  5. Diversity-satisfaction correlation — does more diversity = more sat?

Usage:
    python evaluate.py
    python evaluate.py --n_episodes 1000
"""

import os
import argparse
import numpy as np
from stable_baselines3 import PPO, DQN, A2C
from environment import MovieRecEnv


# ─────────────────────────────────────────
# 1. Args
# ─────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_episodes",  type=int, default=1000)
    parser.add_argument("--agents",      nargs="+", default=["ppo", "dqn", "a2c"],
                        choices=["ppo", "dqn", "a2c"])
    parser.add_argument("--data_dir",    type=str, default="./data/processed")
    parser.add_argument("--models_dir",  type=str, default="./models")
    parser.add_argument("--seed",        type=int, default=123)
    return parser.parse_args()


# ─────────────────────────────────────────
# 2. Diversity metrics
# ─────────────────────────────────────────

def compute_ild(movie_ids, movie_embeddings):
    """
    Intra-List Diversity (ILD) — mean pairwise cosine distance between
    all recommended movies in a list.

    ILD = (1 / |L|*(|L|-1)) * sum_{i≠j} (1 - cosine_sim(i, j))

    Range: [0, 1]
      1.0 = all movies completely different (maximum diversity)
      0.0 = all movies identical (no diversity)

    Reference: Ziegler et al., WWW 2005.
    """
    if len(movie_ids) < 2:
        return 0.0

    vecs = np.array([movie_embeddings[mid] for mid in movie_ids
                     if mid in movie_embeddings])
    if len(vecs) < 2:
        return 0.0

    # Cosine similarities (vectors already L2-normalized from BPR)
    # so dot product = cosine similarity
    sim_matrix = vecs @ vecs.T  # (n, n)

    # Get upper triangle (exclude diagonal)
    n = len(vecs)
    total_sim = 0.0
    count     = 0
    for i in range(n):
        for j in range(i + 1, n):
            total_sim += sim_matrix[i, j]
            count += 1

    if count == 0:
        return 0.0

    mean_sim = total_sim / count
    ild      = 1.0 - mean_sim   # distance = 1 - similarity
    return float(np.clip(ild, 0.0, 1.0))


def compute_coverage(all_recommended_sets, catalog_size):
    """
    Catalog coverage — fraction of catalog recommended across all episodes.
    Higher = agent explores more of the catalog.
    """
    all_recommended = set()
    for s in all_recommended_sets:
        all_recommended.update(s)
    return len(all_recommended) / catalog_size


def compute_novelty(movie_ids, movie_popularity):
    """
    Novelty — average negative log popularity of recommended movies.
    Higher = agent recommends less popular (more novel) movies.
    novelty = mean(-log2(popularity(m))) for m in recommendations
    """
    if not movie_ids:
        return 0.0
    scores = []
    for mid in movie_ids:
        pop = movie_popularity.get(mid, 1)
        scores.append(-np.log2(pop + 1e-10))
    return float(np.mean(scores))


# ─────────────────────────────────────────
# 3. Baselines
# ─────────────────────────────────────────

def greedy_action(env):
    sims = env.emb_matrix @ env.user_profile
    for mid in env.recommended_set:
        if mid in env.movie_to_action:
            sims[env.movie_to_action[mid]] = -999
    return int(np.argmax(sims))


def random_action(env):
    return env.action_space.sample()


# ─────────────────────────────────────────
# 4. Core evaluation
# ─────────────────────────────────────────

def evaluate_agent(env, policy_fn, n_episodes, agent_name, movie_popularity):
    """
    Run n_episodes and collect comprehensive metrics including ILD.
    """
    ep_length = env.episode_length

    # Episode-level
    all_rewards        = []
    all_final_sats     = []
    all_unique_genres  = []
    all_unique_movies  = []
    all_rep_counts     = []
    all_mean_divs      = []
    all_mean_ratings   = []
    all_ilds           = []       # ILD per episode
    all_novelties      = []       # novelty per episode
    all_recommended    = []       # set of movies per episode (for coverage)

    # Timestep curves
    sat_sum    = np.zeros(ep_length)
    div_sum    = np.zeros(ep_length)
    reward_sum = np.zeros(ep_length)
    rating_sum = np.zeros(ep_length)
    genre_sum  = np.zeros(ep_length)
    step_count = np.zeros(ep_length)

    # Rating signal
    rated_scores   = []
    unrated_scores = []

    for ep in range(n_episodes):
        obs, info     = env.reset()
        ep_reward     = 0.0
        ep_reps       = 0
        ep_divs       = []
        ep_ratings    = []
        ep_movie_ids  = []   # movies recommended this episode (for ILD)

        for t in range(ep_length):
            action                     = policy_fn(obs, env)
            obs, reward, done, _, info = env.step(action)

            ep_reward += reward
            ep_reps   += int(info["repetition_penalty"] > 0)
            ep_divs.append(info["diversity_bonus"])
            ep_ratings.append(info["rating_score"])
            ep_movie_ids.append(info["movie_id"])

            sat_sum[t]    += info["satisfaction"]
            div_sum[t]    += info["diversity_bonus"]
            reward_sum[t] += reward
            rating_sum[t] += info["rating_score"]
            genre_sum[t]  += info["n_unique_genres"]
            step_count[t] += 1

            movie_id = info["movie_id"]
            if movie_id in env.user_rating_dict:
                rated_scores.append(info["rating_score"])
            else:
                unrated_scores.append(info["rating_score"])

            if done:
                break

        # Compute ILD for this episode
        ild = compute_ild(ep_movie_ids, env.movie_embeddings)

        # Compute novelty for this episode
        novelty = compute_novelty(ep_movie_ids, movie_popularity)

        all_rewards.append(ep_reward)
        all_final_sats.append(info["satisfaction"])
        all_unique_genres.append(info["n_unique_genres"])
        all_unique_movies.append(info["n_unique_movies"])
        all_rep_counts.append(ep_reps)
        all_mean_divs.append(np.mean(ep_divs))
        all_mean_ratings.append(np.mean(ep_ratings))
        all_ilds.append(ild)
        all_novelties.append(novelty)
        all_recommended.append(set(ep_movie_ids))

        if (ep + 1) % 200 == 0:
            print(f"    {agent_name:14s} — ep {ep+1:5d}/{n_episodes} | "
                  f"reward={np.mean(all_rewards):.3f} | "
                  f"sat={np.mean(all_final_sats):.3f} | "
                  f"ILD={np.mean(all_ilds):.3f} | "
                  f"genres={np.mean(all_unique_genres):.1f}")

    # Normalize curves
    safe         = np.maximum(step_count, 1)
    sat_curve    = sat_sum    / safe
    div_curve    = div_sum    / safe
    reward_curve = reward_sum / safe
    rating_curve = rating_sum / safe
    genre_curve  = genre_sum  / safe

    # Coverage
    coverage = compute_coverage(all_recommended, env.n_movies)

    # Diversity-satisfaction correlation
    div_sat_corr = float(np.corrcoef(
        np.array(all_ilds),
        np.array(all_final_sats)
    )[0, 1])

    return {
        "name":           agent_name,
        "n_episodes":     n_episodes,

        # Episode arrays
        "rewards":        np.array(all_rewards),
        "final_sats":     np.array(all_final_sats),
        "unique_genres":  np.array(all_unique_genres),
        "unique_movies":  np.array(all_unique_movies),
        "rep_counts":     np.array(all_rep_counts),
        "mean_divs":      np.array(all_mean_divs),
        "mean_ratings":   np.array(all_mean_ratings),
        "ilds":           np.array(all_ilds),
        "novelties":      np.array(all_novelties),

        # Curves
        "sat_curve":      sat_curve,
        "div_curve":      div_curve,
        "reward_curve":   reward_curve,
        "rating_curve":   rating_curve,
        "genre_curve":    genre_curve,

        # Aggregate
        "coverage":       coverage,
        "div_sat_corr":   div_sat_corr,
        "rated_scores":   np.array(rated_scores),
        "unrated_scores": np.array(unrated_scores),
    }


# ─────────────────────────────────────────
# 5. Print functions
# ─────────────────────────────────────────

def print_summary(res):
    r   = res["rewards"]
    s   = res["final_sats"]
    g   = res["unique_genres"]
    m   = res["unique_movies"]
    rp  = res["rep_counts"]
    ild = res["ilds"]
    nov = res["novelties"]

    print(f"\n  ── {res['name']} ({res['n_episodes']} episodes) ──")
    print(f"  Cumulative reward     : {r.mean():.3f} ± {r.std():.3f}  "
          f"[min={r.min():.2f}  max={r.max():.2f}]")
    print(f"  Final satisfaction    : {s.mean():.3f} ± {s.std():.3f}")
    print(f"  Unique genres/ep      : {g.mean():.1f} ± {g.std():.1f}  "
          f"(out of 18 total)")
    print(f"  Unique movies/ep      : {m.mean():.1f} ± {m.std():.1f}  "
          f"(out of 20 steps)")
    print(f"  Repetitions/ep        : {rp.mean():.2f} ± {rp.std():.2f}")
    print(f"  ── Diversity Metrics ──────────────────────────────────────")
    print(f"  ILD (Intra-List Div)  : {ild.mean():.4f} ± {ild.std():.4f}  "
          f"← KEY METRIC [0=identical, 1=maximally diverse]")
    print(f"  Mean div bonus/ep     : {res['mean_divs'].mean():.3f} ± "
          f"{res['mean_divs'].std():.3f}")
    print(f"  Catalog coverage      : {res['coverage']*100:.1f}%  "
          f"({int(res['coverage']*res['n_episodes'])} unique movies "
          f"across all episodes)")
    print(f"  Novelty score         : {nov.mean():.3f} ± {nov.std():.3f}")
    print(f"  ILD-Sat correlation   : {res['div_sat_corr']:.3f}  "
          f"(ILD predicts satisfaction)")
    print(f"  ── Rating Signal ──────────────────────────────────────────")
    if len(res["rated_scores"]) > 0:
        print(f"  Rated movies score    : {res['rated_scores'].mean():.3f} "
              f"(n={len(res['rated_scores']):,})")
    if len(res["unrated_scores"]) > 0:
        print(f"  Unrated movies score  : {res['unrated_scores'].mean():.3f} "
              f"(n={len(res['unrated_scores']):,})")
    curve = res["sat_curve"]
    print(f"  Sat (1→5→10→15→20)   : "
          f"{curve[0]:.3f}→{curve[4]:.3f}→{curve[9]:.3f}"
          f"→{curve[14]:.3f}→{curve[19]:.3f}")


def print_comparison(results):
    print(f"\n{'='*80}")
    print(f"  EVALUATION COMPARISON ({results[0]['n_episodes']} episodes)")
    print(f"{'='*80}")
    print(f"  {'Agent':14s}  {'Reward':>8s}  {'Sat':>6s}  "
          f"{'ILD':>7s}  {'Genres':>7s}  {'Cover':>6s}  {'Reps':>5s}")
    print(f"  {'-'*72}")
    for res in results:
        print(f"  {res['name']:14s}  "
              f"{res['rewards'].mean():>8.3f}  "
              f"{res['final_sats'].mean():>6.3f}  "
              f"{res['ilds'].mean():>7.4f}  "
              f"{res['unique_genres'].mean():>7.1f}  "
              f"{res['coverage']*100:>5.1f}%  "
              f"{res['rep_counts'].mean():>5.2f}")
    print(f"{'='*80}")


def print_diversity_analysis(results):
    """Dedicated diversity section for report."""
    print(f"\n{'='*80}")
    print(f"  DIVERSITY ANALYSIS")
    print(f"{'='*80}")
    print(f"\n  ILD (Intra-List Diversity) — higher is more diverse:")
    print(f"  {'Agent':14s}  {'ILD Mean':>10s}  {'ILD Std':>8s}  "
          f"{'vs Greedy':>10s}  {'ILD-Sat r':>10s}")
    print(f"  {'-'*60}")

    greedy_ild = next(r["ilds"].mean() for r in results if r["name"] == "Greedy")

    for res in results:
        ild_mean = res["ilds"].mean()
        ild_std  = res["ilds"].std()
        delta    = ild_mean - greedy_ild
        corr     = res["div_sat_corr"]
        print(f"  {res['name']:14s}  {ild_mean:>10.4f}  {ild_std:>8.4f}  "
              f"{delta:>+10.4f}  {corr:>10.3f}")

    print(f"\n  Interpretation:")
    print(f"  ILD measures average pairwise distance between recommended movies.")
    print(f"  Higher ILD = agent recommends more varied content each session.")
    print(f"  ILD-Sat correlation shows whether diversity predicts satisfaction.")

    print(f"\n  Catalog Coverage — fraction of 500 movies recommended:")
    for res in results:
        bar_len = int(res["coverage"] * 30)
        bar     = "█" * bar_len + "░" * (30 - bar_len)
        print(f"  {res['name']:14s}  [{bar}] {res['coverage']*100:.1f}%")


def print_statistical_tests(results):
    print(f"\n  Statistical Tests vs Greedy (Welch's t-test, "
          f"n={results[0]['n_episodes']}):")
    try:
        from scipy import stats
        greedy_r   = next(r["rewards"] for r in results if r["name"] == "Greedy")
        greedy_s   = next(r["final_sats"] for r in results if r["name"] == "Greedy")
        greedy_ild = next(r["ilds"] for r in results if r["name"] == "Greedy")

        for res in results:
            if res["name"] == "Greedy":
                continue
            t_r,   p_r   = stats.ttest_ind(res["rewards"],     greedy_r,   equal_var=False)
            t_s,   p_s   = stats.ttest_ind(res["final_sats"],  greedy_s,   equal_var=False)
            t_ild, p_ild = stats.ttest_ind(res["ilds"],        greedy_ild, equal_var=False)

            sig_r   = "***" if p_r   < 0.001 else "**" if p_r   < 0.01 else "*" if p_r   < 0.05 else "ns"
            sig_s   = "***" if p_s   < 0.001 else "**" if p_s   < 0.01 else "*" if p_s   < 0.05 else "ns"
            sig_ild = "***" if p_ild < 0.001 else "**" if p_ild < 0.01 else "*" if p_ild < 0.05 else "ns"

            delta_r   = res["rewards"].mean()    - greedy_r.mean()
            delta_ild = res["ilds"].mean()       - greedy_ild.mean()

            print(f"\n    {res['name']}:")
            print(f"      Reward : t={t_r:+7.3f} p={p_r:.4f} {sig_r}  "
                  f"Δ={delta_r:+.3f} ({100*delta_r/abs(greedy_r.mean()):+.1f}%)")
            print(f"      Sat    : t={t_s:+7.3f} p={p_s:.4f} {sig_s}")
            print(f"      ILD    : t={t_ild:+7.3f} p={p_ild:.4f} {sig_ild}  "
                  f"Δ={delta_ild:+.4f}")
    except ImportError:
        print("    scipy not available")


def print_satisfaction_table(results):
    print(f"\n  Satisfaction Curve (mean across {results[0]['n_episodes']} episodes):")
    print(f"  {'Step':>4s}", end="")
    for res in results:
        print(f"  {res['name']:>12s}", end="")
    print()
    print("  " + "-" * (6 + 14 * len(results)))
    for t in range(20):
        print(f"  {t+1:>4d}", end="")
        for res in results:
            if t < len(res["sat_curve"]):
                print(f"  {res['sat_curve'][t]:>12.3f}", end="")
        print()


# ─────────────────────────────────────────
# 6. Main
# ─────────────────────────────────────────

def main():
    args = parse_args()

    print(f"\n{'='*62}")
    print(f"  COMPREHENSIVE EVALUATION WITH DIVERSITY METRICS")
    print(f"{'='*62}")
    print(f"  Episodes    : {args.n_episodes}")
    print(f"  Eval seed   : {args.seed}  (≠ training seed=42)")
    print(f"  Key metric  : ILD (Intra-List Diversity)")

    env = MovieRecEnv(data_dir=args.data_dir, seed=args.seed)

    # Build movie popularity dict for novelty metric
    # (number of users who rated each movie — proxy for popularity)
    user_ratings_all = env.user_ratings
    movie_popularity = {}
    for uid, ratings in user_ratings_all.items():
        for mid in ratings:
            movie_popularity[mid] = movie_popularity.get(mid, 0) + 1
    # Normalize by max
    max_pop = max(movie_popularity.values()) if movie_popularity else 1
    movie_popularity = {mid: cnt/max_pop for mid, cnt in movie_popularity.items()}

    all_results = []

    # ── Greedy ───────────────────────────────────────────────────────────
    print(f"\n  Evaluating Greedy ...")
    res = evaluate_agent(env, lambda obs, e: greedy_action(e),
                         args.n_episodes, "Greedy", movie_popularity)
    print_summary(res)
    all_results.append(res)

    # ── Random ───────────────────────────────────────────────────────────
    print(f"\n  Evaluating Random ...")
    res = evaluate_agent(env, lambda obs, e: random_action(e),
                         args.n_episodes, "Random", movie_popularity)
    print_summary(res)
    all_results.append(res)

    # ── RL agents ────────────────────────────────────────────────────────
    agent_classes = {"ppo": PPO, "dqn": DQN, "a2c": A2C}
    agent_labels  = {"ppo": "PPO", "dqn": "Double DQN", "a2c": "A2C"}

    for name in args.agents:
        model_path = os.path.join(args.models_dir, name, "model.zip")
        if not os.path.exists(model_path):
            print(f"\n  ⚠ {name.upper()} not found — skipping")
            continue

        print(f"\n  Evaluating {agent_labels[name]} ...")
        model = agent_classes[name].load(model_path, env=env)

        def make_policy(m):
            def fn(obs, e):
                action, _ = m.predict(obs, deterministic=False)
                return int(action)
            return fn

        res = evaluate_agent(env, make_policy(model),
                             args.n_episodes, agent_labels[name],
                             movie_popularity)
        print_summary(res)
        all_results.append(res)

    # ── Summary tables ───────────────────────────────────────────────────
    print_comparison(all_results)
    print_diversity_analysis(all_results)
    print_statistical_tests(all_results)
    print_satisfaction_table(all_results)

    # ── Save ─────────────────────────────────────────────────────────────
    out_path = os.path.join(args.models_dir, "eval_results.npy")
    np.save(out_path, all_results)
    print(f"\n  Results saved → {out_path}")
    print(f"  Next: python plot.py")


if __name__ == "__main__":
    main()
