"""
plot.py
=======
Generate all figures for the report.

Produces 6 publication-quality plots:
  1. Cumulative reward comparison (bar chart with error bars)
  2. Satisfaction over timesteps (line chart) — the main result
  3. Genre diversity per agent (bar chart)
  4. Training learning curves (from training_metrics.npy)
  5. Reward distribution (box plot)
  6. Diversity-satisfaction correlation (scatter)

All saved to ./figures/ as high-res PNG files.

Usage:
    python plot.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ─────────────────────────────────────────
# 1. Config
# ─────────────────────────────────────────

FIGURES_DIR  = "./figures"
EVAL_PATH    = "./models/eval_results.npy"
TRAIN_PATH   = "./models/training_metrics.npy"

# Consistent colors for all plots
COLORS = {
    "PPO":        "#2196F3",   # blue
    "A2C":        "#4CAF50",   # green
    "Double DQN": "#FF5722",   # deep orange
    "Random":     "#9E9E9E",   # grey
    "Greedy":     "#F44336",   # red
}

LINESTYLES = {
    "PPO":        "-",
    "A2C":        "-",
    "Double DQN": "--",
    "Random":     ":",
    "Greedy":     "--",
}

MARKERS = {
    "PPO":        "o",
    "A2C":        "s",
    "Double DQN": "^",
    "Random":     "D",
    "Greedy":     "x",
}

# Plot style
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.labelsize":   12,
    "legend.fontsize":  10,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
})


# ─────────────────────────────────────────
# 2. Load data
# ─────────────────────────────────────────

def load_data():
    eval_results = np.load(EVAL_PATH,  allow_pickle=True).tolist()
    try:
        train_metrics = np.load(TRAIN_PATH, allow_pickle=True).item()
    except Exception:
        train_metrics = {}
        print("  Warning: training_metrics.npy not found — skipping learning curves")
    return eval_results, train_metrics


def get_agent_order():
    """Consistent ordering across all plots."""
    return ["Greedy", "Random", "Double DQN", "PPO", "A2C"]


def find_result(results, name):
    return next((r for r in results if r["name"] == name), None)


# ─────────────────────────────────────────
# 3. Plot 1 — Cumulative reward bar chart
# ─────────────────────────────────────────

def plot_reward_comparison(results, out_dir):
    fig, ax = plt.subplots(figsize=(9, 5))

    order  = get_agent_order()
    names  = []
    means  = []
    stds   = []
    colors = []

    for name in order:
        res = find_result(results, name)
        if res is None:
            continue
        names.append(name)
        means.append(res["rewards"].mean())
        stds.append(res["rewards"].std())
        colors.append(COLORS.get(name, "#607D8B"))

    x    = np.arange(len(names))
    bars = ax.bar(x, means, yerr=stds, capsize=5,
                  color=colors, alpha=0.85, edgecolor="white",
                  linewidth=1.5, error_kw=dict(elinewidth=1.5, ecolor="#333"))

    # Value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + std + 0.1,
                f"{mean:.2f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Reference lines
    ax.axhline(y=means[order.index("Greedy") if "Greedy" in order else 0],
               color=COLORS["Greedy"], linestyle="--", alpha=0.4, linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Mean Cumulative Reward (± std)")
    ax.set_title("Cumulative Reward Comparison\n(1000 evaluation episodes per agent)")
    ax.set_ylim(0, max(means) * 1.18)

    # Significance annotations
    greedy_mean = means[names.index("Greedy")] if "Greedy" in names else 0
    for i, (name, mean) in enumerate(zip(names, means)):
        if name in ["PPO", "A2C"]:
            ax.text(i, mean + stds[i] + 0.55, "***",
                    ha="center", fontsize=11, color="#333")

    fig.tight_layout()
    path = os.path.join(out_dir, "fig1_reward_comparison.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────
# 4. Plot 2 — Satisfaction over timesteps
# ─────────────────────────────────────────

def plot_satisfaction_curves(results, out_dir):
    fig, ax = plt.subplots(figsize=(10, 5.5))

    order = get_agent_order()
    steps = np.arange(1, 21)

    for name in order:
        res = find_result(results, name)
        if res is None:
            continue
        curve = res["sat_curve"]
        if len(curve) < 20:
            continue

        ax.plot(steps, curve,
                label=name,
                color=COLORS.get(name, "#607D8B"),
                linestyle=LINESTYLES.get(name, "-"),
                marker=MARKERS.get(name, "o"),
                markersize=4,
                linewidth=2.0,
                alpha=0.9)

    # Shade the gap between best RL and greedy
    ppo_res    = find_result(results, "PPO")
    greedy_res = find_result(results, "Greedy")
    if ppo_res and greedy_res:
        ax.fill_between(steps,
                        greedy_res["sat_curve"][:20],
                        ppo_res["sat_curve"][:20],
                        alpha=0.07, color=COLORS["PPO"],
                        label="_nolegend_")

    ax.set_xlabel("Recommendation Step (within episode)")
    ax.set_ylabel("Mean User Satisfaction Score")
    ax.set_title("User Satisfaction Over Time\n"
                 "(mean across 1000 episodes — key result)")
    ax.set_xlim(1, 20)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(steps)

    # Annotate final values
    for name in order:
        res = find_result(results, name)
        if res is None or len(res["sat_curve"]) < 20:
            continue
        final = res["sat_curve"][19]
        ax.annotate(f"{final:.3f}",
                    xy=(20, final),
                    xytext=(20.3, final),
                    fontsize=8.5,
                    color=COLORS.get(name, "#607D8B"),
                    va="center")

    ax.legend(loc="lower left", framealpha=0.9)
    fig.tight_layout()
    path = os.path.join(out_dir, "fig2_satisfaction_curves.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────
# 5. Plot 3 — Genre diversity bar chart
# ─────────────────────────────────────────

def plot_genre_diversity(results, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    order = get_agent_order()

    # Left: unique genres per episode
    ax = axes[0]
    names, means_g, stds_g, colors = [], [], [], []
    for name in order:
        res = find_result(results, name)
        if res is None:
            continue
        names.append(name)
        means_g.append(res["unique_genres"].mean())
        stds_g.append(res["unique_genres"].std())
        colors.append(COLORS.get(name, "#607D8B"))

    x    = np.arange(len(names))
    bars = ax.bar(x, means_g, yerr=stds_g, capsize=4,
                  color=colors, alpha=0.85, edgecolor="white",
                  linewidth=1.5, error_kw=dict(elinewidth=1.5))
    for bar, mean in zip(bars, means_g):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.2,
                f"{mean:.1f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Mean Unique Genres per Episode")
    ax.set_title("Genre Diversity per Episode\n(out of 18 total genres)")
    ax.set_ylim(0, 18)

    # Right: mean diversity bonus
    ax = axes[1]
    names2, means_d, stds_d, colors2 = [], [], [], []
    for name in order:
        res = find_result(results, name)
        if res is None:
            continue
        names2.append(name)
        means_d.append(res["mean_divs"].mean())
        stds_d.append(res["mean_divs"].std())
        colors2.append(COLORS.get(name, "#607D8B"))

    x2   = np.arange(len(names2))
    bars = ax.bar(x2, means_d, yerr=stds_d, capsize=4,
                  color=colors2, alpha=0.85, edgecolor="white",
                  linewidth=1.5, error_kw=dict(elinewidth=1.5))
    for bar, mean in zip(bars, means_d):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.003,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x2)
    ax.set_xticklabels(names2, fontsize=9)
    ax.set_ylabel("Mean Embedding Diversity Bonus")
    ax.set_title("Embedding Diversity Score\n(1.0 = maximally diverse)")
    ax.set_ylim(0, 0.65)

    fig.suptitle("Recommendation Diversity Metrics", fontsize=13, y=1.01)
    fig.tight_layout()
    path = os.path.join(out_dir, "fig3_diversity.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────
# 6. Plot 4 — Training learning curves
# ─────────────────────────────────────────

def smooth(values, window=50):
    """Rolling mean smoothing."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_training_curves(train_metrics, out_dir):
    if not train_metrics:
        print("  Skipping training curves (no data)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    agent_map = {
        "ppo": ("PPO", COLORS["PPO"]),
        "a2c": ("A2C", COLORS["A2C"]),
        "dqn": ("Double DQN", COLORS["Double DQN"]),
    }

    window = 100

    # Left: reward curves
    ax = axes[0]
    for key, (label, color) in agent_map.items():
        if key not in train_metrics:
            continue
        rewards = train_metrics[key]["rewards"]
        smoothed = smooth(rewards, window)
        episodes = np.arange(window, len(rewards) + 1)
        ax.plot(episodes, smoothed,
                label=label, color=color, linewidth=1.8, alpha=0.9)
        # Raw values faint
        ax.plot(np.arange(1, len(rewards)+1), rewards,
                color=color, alpha=0.1, linewidth=0.5)

    ax.set_xlabel("Training Episode")
    ax.set_ylabel(f"Cumulative Reward (rolling mean, window={window})")
    ax.set_title("Training Learning Curves — Reward")
    ax.legend()

    # Right: satisfaction curves
    ax = axes[1]
    for key, (label, color) in agent_map.items():
        if key not in train_metrics:
            continue
        sats = train_metrics[key]["sats"]
        smoothed = smooth(sats, window)
        episodes = np.arange(window, len(sats) + 1)
        ax.plot(episodes, smoothed,
                label=label, color=color, linewidth=1.8, alpha=0.9)
        ax.plot(np.arange(1, len(sats)+1), sats,
                color=color, alpha=0.1, linewidth=0.5)

    ax.set_xlabel("Training Episode")
    ax.set_ylabel(f"Final Satisfaction (rolling mean, window={window})")
    ax.set_title("Training Learning Curves — Satisfaction")
    ax.legend()

    fig.suptitle("RL Agent Training Curves", fontsize=13, y=1.01)
    fig.tight_layout()
    path = os.path.join(out_dir, "fig4_training_curves.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────
# 7. Plot 5 — Reward distribution box plot
# ─────────────────────────────────────────

def plot_reward_distribution(results, out_dir):
    fig, ax = plt.subplots(figsize=(10, 5))

    order = get_agent_order()
    data  = []
    names = []
    colors = []

    for name in order:
        res = find_result(results, name)
        if res is None:
            continue
        data.append(res["rewards"])
        names.append(name)
        colors.append(COLORS.get(name, "#607D8B"))

    bp = ax.boxplot(data,
                    patch_artist=True,
                    notch=False,
                    medianprops=dict(color="white", linewidth=2),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5),
                    flierprops=dict(marker=".", markersize=3, alpha=0.4))

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Cumulative Reward per Episode")
    ax.set_title("Reward Distribution per Agent\n"
                 "(1000 episodes — shows consistency and range)")

    # Add mean markers
    for i, d in enumerate(data):
        ax.scatter(i + 1, np.mean(d), marker="D",
                   color="white", s=40, zorder=5,
                   edgecolors=colors[i], linewidth=1.5)

    # Legend for mean marker
    mean_patch = mpatches.Patch(facecolor="white",
                                edgecolor="#333", label="◈ Mean")
    ax.legend(handles=[mean_patch], loc="upper left")

    fig.tight_layout()
    path = os.path.join(out_dir, "fig5_reward_distribution.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────
# 8. Plot 6 — Satisfaction vs diversity
# ─────────────────────────────────────────

def plot_diversity_satisfaction(results, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5.5))

    order = get_agent_order()

    for name in order:
        res = find_result(results, name)
        if res is None:
            continue

        divs = res["mean_divs"]
        sats = res["final_sats"]
        color = COLORS.get(name, "#607D8B")

        # Subsample for readability (max 200 points)
        n = len(divs)
        if n > 200:
            idx = np.random.choice(n, 200, replace=False)
            divs = divs[idx]
            sats = sats[idx]

        ax.scatter(divs, sats,
                   label=f"{name} (r={res['div_sat_corr']:.2f})",
                   color=color, alpha=0.35, s=18,
                   edgecolors="none")

        # Trend line
        if len(divs) > 10:
            z = np.polyfit(divs, sats, 1)
            p = np.poly1d(z)
            x_line = np.linspace(divs.min(), divs.max(), 100)
            ax.plot(x_line, p(x_line), color=color, linewidth=2, alpha=0.9)

    ax.set_xlabel("Mean Diversity Bonus per Episode")
    ax.set_ylabel("Final Satisfaction Score")
    ax.set_title("Diversity vs Satisfaction Correlation\n"
                 "(each point = one episode, lines = linear trend)")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_xlim(0.25, 0.65)
    ax.set_ylim(-0.05, 1.05)

    fig.tight_layout()
    path = os.path.join(out_dir, "fig6_diversity_satisfaction.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────
# 9. Summary figure (all 4 key plots in one)
# ─────────────────────────────────────────

def plot_summary_grid(results, train_metrics, out_dir):
    """
    2×2 grid of the 4 most important plots for the report.
    Use this as the main figure.
    """
    fig = plt.figure(figsize=(16, 11))
    gs  = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.30)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    order = get_agent_order()
    steps = np.arange(1, 21)

    # ── Top left: Reward bar chart ───────────────────────────────────
    names, means, stds, colors = [], [], [], []
    for name in order:
        res = find_result(results, name)
        if res is None:
            continue
        names.append(name)
        means.append(res["rewards"].mean())
        stds.append(res["rewards"].std())
        colors.append(COLORS.get(name, "#607D8B"))

    x    = np.arange(len(names))
    bars = ax1.bar(x, means, yerr=stds, capsize=4,
                   color=colors, alpha=0.85, edgecolor="white",
                   error_kw=dict(elinewidth=1.5))
    for bar, mean in zip(bars, means):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.3,
                 f"{mean:.2f}", ha="center", va="bottom", fontsize=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=8)
    ax1.set_ylabel("Mean Cumulative Reward")
    ax1.set_title("(a) Cumulative Reward Comparison")
    ax1.set_ylim(0, max(means) * 1.2)

    # ── Top right: Satisfaction curves ───────────────────────────────
    for name in order:
        res = find_result(results, name)
        if res is None or len(res["sat_curve"]) < 20:
            continue
        ax2.plot(steps, res["sat_curve"],
                 label=name,
                 color=COLORS.get(name, "#607D8B"),
                 linestyle=LINESTYLES.get(name, "-"),
                 marker=MARKERS.get(name, "o"),
                 markersize=3, linewidth=1.8, alpha=0.9)

    ax2.set_xlabel("Recommendation Step")
    ax2.set_ylabel("Mean Satisfaction")
    ax2.set_title("(b) User Satisfaction Over Time")
    ax2.set_xlim(1, 20)
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=8, loc="lower left")

    # ── Bottom left: Training curves ─────────────────────────────────
    if train_metrics:
        window = 100
        agent_map = {
            "ppo": ("PPO", COLORS["PPO"]),
            "a2c": ("A2C", COLORS["A2C"]),
            "dqn": ("Double DQN", COLORS["Double DQN"]),
        }
        for key, (label, color) in agent_map.items():
            if key not in train_metrics:
                continue
            rewards  = train_metrics[key]["rewards"]
            smoothed = smooth(rewards, window)
            episodes = np.arange(window, len(rewards) + 1)
            ax3.plot(episodes, smoothed,
                     label=label, color=color, linewidth=1.8)
        ax3.set_xlabel("Training Episode")
        ax3.set_ylabel("Reward (rolling mean)")
        ax3.set_title("(c) Training Learning Curves")
        ax3.legend(fontsize=8)
    else:
        ax3.text(0.5, 0.5, "Training data\nnot available",
                 ha="center", va="center", transform=ax3.transAxes)
        ax3.set_title("(c) Training Learning Curves")

    # ── Bottom right: Box plot ────────────────────────────────────────
    data_box = []
    names_box = []
    colors_box = []
    for name in order:
        res = find_result(results, name)
        if res is None:
            continue
        data_box.append(res["rewards"])
        names_box.append(name)
        colors_box.append(COLORS.get(name, "#607D8B"))

    bp = ax4.boxplot(data_box, patch_artist=True, notch=False,
                     medianprops=dict(color="white", linewidth=2),
                     whiskerprops=dict(linewidth=1.2),
                     capprops=dict(linewidth=1.2),
                     flierprops=dict(marker=".", markersize=2, alpha=0.3))
    for patch, color in zip(bp["boxes"], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax4.set_xticklabels(names_box, fontsize=8)
    ax4.set_ylabel("Cumulative Reward")
    ax4.set_title("(d) Reward Distribution")

    fig.suptitle(
        "RL-Based Movie Recommendation — Key Results\n"
        "(1000 evaluation episodes, seed=123)",
        fontsize=14, y=1.01
    )

    path = os.path.join(out_dir, "fig0_summary.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────
# 10. Main
# ─────────────────────────────────────────

def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  GENERATING FIGURES")
    print(f"{'='*55}")
    print(f"  Output dir: {FIGURES_DIR}/")

    eval_results, train_metrics = load_data()

    print(f"\n  Agents in eval results:")
    for res in eval_results:
        print(f"    {res['name']:14s} — {res['n_episodes']} episodes, "
              f"reward={res['rewards'].mean():.3f}")

    print(f"\n  Generating plots ...")

    plot_reward_comparison(eval_results, FIGURES_DIR)
    plot_satisfaction_curves(eval_results, FIGURES_DIR)
    plot_genre_diversity(eval_results, FIGURES_DIR)
    plot_training_curves(train_metrics, FIGURES_DIR)
    plot_reward_distribution(eval_results, FIGURES_DIR)
    plot_diversity_satisfaction(eval_results, FIGURES_DIR)
    plot_summary_grid(eval_results, train_metrics, FIGURES_DIR)

    print(f"\n{'='*55}")
    print(f"  ALL FIGURES SAVED TO {FIGURES_DIR}/")
    print(f"{'='*55}")
    print(f"  fig0_summary.png          ← main 2×2 grid for report")
    print(f"  fig1_reward_comparison.png")
    print(f"  fig2_satisfaction_curves.png  ← most important result")
    print(f"  fig3_diversity.png")
    print(f"  fig4_training_curves.png")
    print(f"  fig5_reward_distribution.png")
    print(f"  fig6_diversity_satisfaction.png")


if __name__ == "__main__":
    main()
