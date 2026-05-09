"""
evaluate.py
===========
Comprehensive evaluation of trained PPO, Double DQN, and A2C agents.

Collects everything needed for plots and report:
  - Per-episode reward, satisfaction, diversity, genres, repetitions
  - Full 20-timestep satisfaction and diversity curves
  - Full 20-timestep reward curves
  - Per-episode reward distributions (for histogram)
  - Diversity-satisfaction correlation
  - Statistical significance vs greedy (Welch's t-test)
  - Rating score breakdown (rated vs unrated movies)

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
    parser.add_argument("--n_episodes",  type=int,   default=1000)
    parser.add_argument("--agents",      nargs="+",  default=["ppo", "dqn", "a2c"],
                        choices=["ppo", "dqn", "a2c"])
    parser.add_argument("--data_dir",    type=str,   default="./data/processed")
    parser.add_argument("--models_dir",  type=str,   default="./models")
    parser.add_argument("--seed",        type=int,   default=123)
    return parser.parse_args()


# ─────────────────────────────────────────
# 2. Baselines
# ─────────────────────────────────────────

def greedy_action(env):
    """Highest BPR similarity, never repeats."""
    sims = env.emb_matrix @ env.user_profile
    for mid in env.recommended_set:
        if mid in env.movie_to_action:
            sims[env.movie_to_action[mid]] = -999
    return int(np.argmax(sims))


def random_action(env):
    return env.action_space.sample()


# ─────────────────────────────────────────
# 3. Core evaluation function
# ─────────────────────────────────────────

def evaluate_agent(env, policy_fn, n_episodes, agent_name):
    """
    Run n_episodes and collect comprehensive metrics.

    Returns dict containing:
      Episode-level:
        rewards          (n_episodes,)  total reward per episode
        final_sats       (n_episodes,)  final satisfaction per episode
        unique_genres    (n_episodes,)  unique genres recommended
        unique_movies    (n_episodes,)  unique movies recommended
        rep_counts       (n_episodes,)  repetitions per episode
        mean_divs        (n_episodes,)  mean diversity bonus per episode
        mean_ratings     (n_episodes,)  mean rating score per episode

      Timestep-level (averaged across episodes):
        sat_curve        (episode_length,)  mean satisfaction at each step
        div_curve        (episode_length,)  mean diversity at each step
        reward_curve     (episode_length,)  mean reward at each step
        rating_curve     (episode_length,)  mean rating score at each step
        genre_curve      (episode_length,)  mean cumulative genres at each step

      Aggregate:
        div_sat_corr     float  correlation between diversity and satisfaction
        rated_scores     list   rating scores when movie was actually rated
        unrated_scores   list   rating scores when movie was estimated
    """
    ep_length = env.episode_length

    # Episode accumulators
    all_rewards       = []
    all_final_sats    = []
    all_unique_genres = []
    all_unique_movies = []
    all_rep_counts    = []
    all_mean_divs     = []
    all_mean_ratings  = []

    # Timestep accumulators
    sat_sum    = np.zeros(ep_length)
    div_sum    = np.zeros(ep_length)
    reward_sum = np.zeros(ep_length)
    rating_sum = np.zeros(ep_length)
    genre_sum  = np.zeros(ep_length)
    step_count = np.zeros(ep_length)

    # Rating signal quality
    rated_scores   = []
    unrated_scores = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        ep_reward  = 0.0
        ep_reps    = 0
        ep_divs    = []
        ep_ratings = []

        for t in range(ep_length):
            action                     = policy_fn(obs, env)
            obs, reward, done, _, info = env.step(action)

            ep_reward += reward
            ep_reps   += int(info["repetition_penalty"] > 0)
            ep_divs.append(info["diversity_bonus"])
            ep_ratings.append(info["rating_score"])

            # Timestep curves
            sat_sum[t]    += info["satisfaction"]
            div_sum[t]    += info["diversity_bonus"]
            reward_sum[t] += reward
            rating_sum[t] += info["rating_score"]
            genre_sum[t]  += info["n_unique_genres"]
            step_count[t] += 1

            # Rating signal breakdown
            movie_id = info["movie_id"]
            if movie_id in env.user_rating_dict:
                rated_scores.append(info["rating_score"])
            else:
                unrated_scores.append(info["rating_score"])

            if done:
                break

        all_rewards.append(ep_reward)
        all_final_sats.append(info["satisfaction"])
        all_unique_genres.append(info["n_unique_genres"])
        all_unique_movies.append(info["n_unique_movies"])
        all_rep_counts.append(ep_reps)
        all_mean_divs.append(np.mean(ep_divs))
        all_mean_ratings.append(np.mean(ep_ratings))

        if (ep + 1) % 200 == 0:
            print(f"    {agent_name:14s} — ep {ep+1:5d}/{n_episodes} | "
                  f"reward={np.mean(all_rewards):.3f} | "
                  f"sat={np.mean(all_final_sats):.3f} | "
                  f"reps={np.mean(all_rep_counts):.2f} | "
                  f"genres={np.mean(all_unique_genres):.1f}")

    # Normalize timestep curves
    safe = np.maximum(step_count, 1)
    sat_curve    = sat_sum    / safe
    div_curve    = div_sum    / safe
    reward_curve = reward_sum / safe
    rating_curve = rating_sum / safe
    genre_curve  = genre_sum  / safe

    # Diversity-satisfaction correlation
    div_sat_corr = float(np.corrcoef(
        np.array(all_mean_divs),
        np.array(all_final_sats)
    )[0, 1])

    return {
        # Identity
        "name":           agent_name,
        "n_episodes":     n_episodes,

        # Episode-level arrays
        "rewards":        np.array(all_rewards),
        "final_sats":     np.array(all_final_sats),
        "unique_genres":  np.array(all_unique_genres),
        "unique_movies":  np.array(all_unique_movies),
        "rep_counts":     np.array(all_rep_counts),
        "mean_divs":      np.array(all_mean_divs),
        "mean_ratings":   np.array(all_mean_ratings),

        # Timestep curves
        "sat_curve":      sat_curve,
        "div_curve":      div_curve,
        "reward_curve":   reward_curve,
        "rating_curve":   rating_curve,
        "genre_curve":    genre_curve,

        # Aggregate
        "div_sat_corr":   div_sat_corr,
        "rated_scores":   np.array(rated_scores),
        "unrated_scores": np.array(unrated_scores),
    }


# ─────────────────────────────────────────
# 4. Print summary
# ─────────────────────────────────────────

def print_summary(res):
    r  = res["rewards"]
    s  = res["final_sats"]
    g  = res["unique_genres"]
    m  = res["unique_movies"]
    rp = res["rep_counts"]
    d  = res["mean_divs"]

    print(f"\n  ── {res['name']} ({res['n_episodes']} episodes) ──")
    print(f"  Cumulative reward   : {r.mean():.3f} ± {r.std():.3f}  "
          f"[min={r.min():.2f}  max={r.max():.2f}]")
    print(f"  Final satisfaction  : {s.mean():.3f} ± {s.std():.3f}")
    print(f"  Unique genres/ep    : {g.mean():.1f} ± {g.std():.1f}")
    print(f"  Unique movies/ep    : {m.mean():.1f} ± {m.std():.1f}  (out of 20)")
    print(f"  Repetitions/ep      : {rp.mean():.2f} ± {rp.std():.2f}")
    print(f"  Mean diversity/ep   : {d.mean():.3f} ± {d.std():.3f}")
    print(f"  Div-Sat correlation : {res['div_sat_corr']:.3f}  "
          f"(higher = diversity drives satisfaction)")

    # Rating signal
    if len(res["rated_scores"]) > 0:
        print(f"  Rated movie score   : {res['rated_scores'].mean():.3f} "
              f"(n={len(res['rated_scores']):,})")
    if len(res["unrated_scores"]) > 0:
        print(f"  Unrated movie score : {res['unrated_scores'].mean():.3f} "
              f"(n={len(res['unrated_scores']):,})")

    # Satisfaction curve
    curve = res["sat_curve"]
    print(f"  Sat curve (steps 1→5→10→15→20): "
          f"{curve[0]:.3f} → {curve[4]:.3f} → {curve[9]:.3f} "
          f"→ {curve[14]:.3f} → {curve[19]:.3f}")


def print_comparison(results):
    print(f"\n{'='*72}")
    print(f"  EVALUATION COMPARISON ({results[0]['n_episodes']} episodes each)")
    print(f"{'='*72}")
    print(f"  {'Agent':14s}  {'Reward':>8s}  {'Sat':>6s}  "
          f"{'Genres':>7s}  {'Movies':>7s}  {'Reps':>5s}  {'Div-Sat':>8s}")
    print(f"  {'-'*68}")
    for res in results:
        r  = res["rewards"].mean()
        s  = res["final_sats"].mean()
        g  = res["unique_genres"].mean()
        m  = res["unique_movies"].mean()
        rp = res["rep_counts"].mean()
        c  = res["div_sat_corr"]
        print(f"  {res['name']:14s}  {r:>8.3f}  {s:>6.3f}  "
              f"{g:>7.1f}  {m:>7.1f}  {rp:>5.2f}  {c:>8.3f}")
    print(f"{'='*72}")


def print_statistical_tests(results):
    print(f"\n  Statistical significance vs Greedy (Welch's t-test, n={results[0]['n_episodes']}):")
    try:
        from scipy import stats
        greedy_r = next(r["rewards"] for r in results if r["name"] == "Greedy")
        greedy_s = next(r["final_sats"] for r in results if r["name"] == "Greedy")

        for res in results:
            if res["name"] == "Greedy":
                continue
            # Reward test
            t_r, p_r = stats.ttest_ind(res["rewards"], greedy_r, equal_var=False)
            # Satisfaction test
            t_s, p_s = stats.ttest_ind(res["final_sats"], greedy_s, equal_var=False)

            sig_r = "***" if p_r < 0.001 else "**" if p_r < 0.01 else "*" if p_r < 0.05 else "ns"
            sig_s = "***" if p_s < 0.001 else "**" if p_s < 0.01 else "*" if p_s < 0.05 else "ns"

            delta  = res["rewards"].mean() - greedy_r.mean()
            pct    = 100 * delta / greedy_r.mean()

            print(f"    {res['name']:14s}: "
                  f"reward t={t_r:+7.3f} p={p_r:.4f} {sig_r}  |  "
                  f"sat t={t_s:+7.3f} p={p_s:.4f} {sig_s}  |  "
                  f"Δreward={delta:+.3f} ({pct:+.1f}%)")
    except ImportError:
        print("    scipy not available")


def print_satisfaction_table(results):
    """Print full 20-step satisfaction table for all agents."""
    print(f"\n  Satisfaction Curve — All Agents (mean across {results[0]['n_episodes']} episodes):")
    print(f"  {'Step':>4s}", end="")
    for res in results:
        print(f"  {res['name']:>12s}", end="")
    print()
    print("  " + "-" * (6 + 14 * len(results)))

    ep_len = len(results[0]["sat_curve"])
    for t in range(ep_len):
        print(f"  {t+1:>4d}", end="")
        for res in results:
            if t < len(res["sat_curve"]):
                print(f"  {res['sat_curve'][t]:>12.3f}", end="")
        print()


# ─────────────────────────────────────────
# 5. Main
# ─────────────────────────────────────────

def main():
    args = parse_args()

    print(f"\n{'='*62}")
    print(f"  COMPREHENSIVE EVALUATION")
    print(f"{'='*62}")
    print(f"  Episodes per agent : {args.n_episodes}")
    print(f"  Agents             : {args.agents}")
    print(f"  Eval seed          : {args.seed}  (different from training seed=42)")

    env = MovieRecEnv(data_dir=args.data_dir, seed=args.seed)

    all_results = []

    # ── Greedy ───────────────────────────────────────────────────────────
    print(f"\n  Evaluating Greedy ...")
    res = evaluate_agent(env, lambda obs, e: greedy_action(e),
                         args.n_episodes, "Greedy")
    print_summary(res)
    all_results.append(res)

    # ── Random ───────────────────────────────────────────────────────────
    print(f"\n  Evaluating Random ...")
    res = evaluate_agent(env, lambda obs, e: random_action(e),
                         args.n_episodes, "Random")
    print_summary(res)
    all_results.append(res)

    # ── RL agents ────────────────────────────────────────────────────────
    agent_classes = {"ppo": PPO, "dqn": DQN, "a2c": A2C}
    agent_labels  = {"ppo": "PPO", "dqn": "Double DQN", "a2c": "A2C"}

    for name in args.agents:
        model_path = os.path.join(args.models_dir, name, "model.zip")
        if not os.path.exists(model_path):
            print(f"\n  ⚠ {name.upper()} not found at {model_path} — skipping")
            continue

        print(f"\n  Evaluating {agent_labels[name]} ...")
        model = agent_classes[name].load(model_path, env=env)

        def make_policy(m):
            def fn(obs, e):
                action, _ = m.predict(obs, deterministic=False)
                return int(action)
            return fn

        res = evaluate_agent(env, make_policy(model),
                             args.n_episodes, agent_labels[name])
        print_summary(res)
        all_results.append(res)

    # ── Comparison tables ────────────────────────────────────────────────
    print_comparison(all_results)
    print_statistical_tests(all_results)
    print_satisfaction_table(all_results)

    # ── Diversity-satisfaction insight ───────────────────────────────────
    print(f"\n  Diversity → Satisfaction Correlation:")
    for res in all_results:
        c = res["div_sat_corr"]
        interp = "strong" if c > 0.5 else "moderate" if c > 0.3 else "weak"
        print(f"    {res['name']:14s}: r={c:.3f}  ({interp} positive relationship)")

    # ── Rating signal breakdown ──────────────────────────────────────────
    print(f"\n  Rating Signal Quality (PPO as example):")
    ppo_res = next((r for r in all_results if r["name"] == "PPO"), None)
    if ppo_res:
        rs = ppo_res["rated_scores"]
        us = ppo_res["unrated_scores"]
        if len(rs) > 0 and len(us) > 0:
            print(f"    Rated movies   : {rs.mean():.3f} ± {rs.std():.3f}  (n={len(rs):,})")
            print(f"    Unrated movies : {us.mean():.3f} ± {us.std():.3f}  (n={len(us):,})")
            print(f"    Gap            : {rs.mean()-us.mean():.3f}  "
                  f"(BPR embeddings predictive of actual ratings)")

    # ── Save ─────────────────────────────────────────────────────────────
    out_path = os.path.join(args.models_dir, "eval_results.npy")
    np.save(out_path, all_results)
    print(f"\n  Results saved → {out_path}")
    print(f"  Next step: python plot.py")


if __name__ == "__main__":
    main()
