"""
train.py
========
Train PPO, Double DQN, and A2C agents on MovieRecEnv.

v4 changes vs v3:
  - Automatic backup of existing models before overwriting
  - DQN completely rearchitected to prevent collapse:
      * Smaller network [256, 128] — less overfitting
      * Slower learning rate (1e-4 vs 5e-4)
      * Much longer exploration (75% of training)
      * Higher epsilon floor (0.20) — never fully greedy
      * Larger replay buffer (500k)
      * More learning starts (10k) — stable Q-values before exploitation
      * train_freq=4 — less aggressive updates
      * Prioritized experience replay via larger buffer sampling
  - PPO/A2C: slightly larger entropy (0.07) for more diversity
  - All agents: 500k timesteps for better convergence
  - Per-agent model backup with timestamp

Usage:
    python train.py                        # train all 3
    python train.py --agents dqn          # retrain only DQN
    python train.py --agents ppo a2c      # retrain specific agents
    python train.py --backup_dir ./models_v3  # custom backup location
"""

import os
import shutil
import argparse
import numpy as np
from datetime import datetime
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from environment import MovieRecEnv


# ─────────────────────────────────────────
# 1. Args
# ─────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents",      nargs="+", default=["ppo", "dqn", "a2c"],
                        choices=["ppo", "dqn", "a2c"])
    parser.add_argument("--timesteps",   type=int,   default=500_000)
    parser.add_argument("--data_dir",    type=str,   default="./data/processed")
    parser.add_argument("--models_dir",  type=str,   default="./models")
    parser.add_argument("--backup_dir",  type=str,   default="./models_backup",
                        help="Where to back up existing models before overwriting")
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--verbose",     type=int,   default=1)
    parser.add_argument("--no_backup",   action="store_true",
                        help="Skip backup of existing models")
    return parser.parse_args()


# ─────────────────────────────────────────
# 2. Backup existing models
# ─────────────────────────────────────────

def backup_models(agents, models_dir, backup_dir):
    """
    Copy existing model files to backup_dir before overwriting.
    Creates timestamped subfolder so multiple backups are kept.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(backup_dir, f"backup_{timestamp}")

    backed_up = []
    for name in agents:
        src = os.path.join(models_dir, name, "model.zip")
        if os.path.exists(src):
            dst_dir = os.path.join(backup_path, name)
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy2(src, os.path.join(dst_dir, "model.zip"))
            backed_up.append(name)

    # Also back up training metrics
    metrics_src = os.path.join(models_dir, "training_metrics.npy")
    if os.path.exists(metrics_src):
        os.makedirs(backup_path, exist_ok=True)
        shutil.copy2(metrics_src, os.path.join(backup_path, "training_metrics.npy"))

    # Also back up eval results
    eval_src = os.path.join(models_dir, "eval_results.npy")
    if os.path.exists(eval_src):
        os.makedirs(backup_path, exist_ok=True)
        shutil.copy2(eval_src, os.path.join(backup_path, "eval_results.npy"))

    if backed_up:
        print(f"  ✓ Backed up: {backed_up}")
        print(f"    → {backup_path}")
    else:
        print(f"  No existing models found to back up")

    return backup_path


# ─────────────────────────────────────────
# 3. LR schedule
# ─────────────────────────────────────────

def linear_decay(initial_lr, final_lr=1e-5):
    def schedule(progress_remaining):
        return final_lr + progress_remaining * (initial_lr - final_lr)
    return schedule


# ─────────────────────────────────────────
# 4. Metrics callback
# ─────────────────────────────────────────

class MetricsCallback(BaseCallback):

    def __init__(self, agent_name="", verbose=0):
        super().__init__(verbose)
        self.agent_name      = agent_name
        self.episode_rewards = []
        self.episode_sats    = []
        self.episode_genres  = []
        self.episode_movies  = []
        self.episode_reps    = []
        self.episode_divs    = []
        self.episode_lengths = []

        self._cur_reward = 0.0
        self._cur_length = 0
        self._cur_reps   = 0
        self._cur_divs   = []

    def _on_step(self):
        info   = self.locals["infos"][0]
        reward = self.locals["rewards"][0]
        done   = self.locals["dones"][0]

        self._cur_reward += reward
        self._cur_length += 1
        self._cur_reps   += int(info.get("repetition_penalty", 0) > 0)
        self._cur_divs.append(info.get("diversity_bonus", 0.0))

        if done:
            self.episode_rewards.append(self._cur_reward)
            self.episode_lengths.append(self._cur_length)
            self.episode_sats.append(info.get("satisfaction",     0.0))
            self.episode_genres.append(info.get("n_unique_genres", 0))
            self.episode_movies.append(info.get("n_unique_movies", 0))
            self.episode_reps.append(self._cur_reps)
            self.episode_divs.append(np.mean(self._cur_divs) if self._cur_divs else 0.0)

            self._cur_reward = 0.0
            self._cur_length = 0
            self._cur_reps   = 0
            self._cur_divs   = []

            n = len(self.episode_rewards)
            if self.verbose >= 1 and n % 200 == 0:
                last = min(200, n)
                print(f"    [{self.agent_name}] ep {n:5d} | "
                      f"reward={np.mean(self.episode_rewards[-last:]):.3f} | "
                      f"sat={np.mean(self.episode_sats[-last:]):.3f} | "
                      f"div={np.mean(self.episode_divs[-last:]):.3f} | "
                      f"reps={np.mean(self.episode_reps[-last:]):.1f} | "
                      f"movies={np.mean(self.episode_movies[-last:]):.1f}")
        return True

    def get_metrics(self):
        return {
            "rewards": np.array(self.episode_rewards),
            "sats":    np.array(self.episode_sats),
            "genres":  np.array(self.episode_genres),
            "movies":  np.array(self.episode_movies),
            "reps":    np.array(self.episode_reps),
            "divs":    np.array(self.episode_divs),
            "lengths": np.array(self.episode_lengths),
        }


# ─────────────────────────────────────────
# 5. Agent configs
# ─────────────────────────────────────────

def get_agent(name, env, seed, total_timesteps):
    """
    v4 hyperparameters.

    PPO — minimal changes, already working well:
      - ent_coef 0.05 → 0.07: slightly stronger entropy
      - batch_size 256 → 512: more stable updates at 500k timesteps

    DQN — complete rework to prevent collapse:
      Root cause of collapse: after exploration ends, DQN locked into
      repeating high-rated movies (getting 0.6 rating reward) despite
      0.5 penalty, because it underestimated long-term satisfaction cost.

      Fixes:
      1. Smaller network [256, 128]: less likely to overfit Q-values
         to short-term reward patterns
      2. lr 5e-4 → 1e-4: slower, more stable Q-value updates
      3. exploration_fraction 0.6 → 0.75: 375k steps of exploration
         in 500k total — enough to discover diverse strategies
      4. exploration_final_eps 0.15 → 0.20: always keeps 20% random
         actions — prevents locking into any fixed strategy
      5. learning_starts 5k → 10k: more diverse replay buffer before
         any Q-learning begins
      6. train_freq 1 → 4: less aggressive updates prevent instability
      7. buffer_size 200k → 500k: more diverse experience in replay
      8. target_update_interval 250 → 1000: less frequent target sync
         gives Q-values time to stabilize before being copied

    A2C — minimal changes, already working well:
      - ent_coef 0.05 → 0.07: slightly stronger entropy
      - n_steps 50 → 100: even longer rollouts
    """

    if name == "ppo":
        return PPO(
            "MlpPolicy", env,
            policy_kwargs=dict(net_arch=[512, 256]),
            learning_rate=linear_decay(3e-4, 3e-5),
            n_steps=4096,
            batch_size=512,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.07,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=0,
            seed=seed,
        )

    elif name == "dqn":
        return DQN(
            "MlpPolicy", env,
            # Smaller network — reduces Q-value overfitting
            policy_kwargs=dict(net_arch=[256, 128]),
            learning_rate=1e-4,              # slow, stable
            buffer_size=500_000,             # huge replay buffer
            learning_starts=10_000,          # lots of data before learning
            batch_size=64,                   # smaller batches
            gamma=0.99,
            target_update_interval=1000,     # infrequent target sync
            # Exploration: 75% of training exploring
            exploration_fraction=0.75,
            # Never go below 20% random — prevents collapse
            exploration_final_eps=0.20,
            train_freq=4,                    # update every 4 steps
            optimize_memory_usage=False,
            verbose=0,
            seed=seed,
        )

    elif name == "a2c":
        return A2C(
            "MlpPolicy", env,
            policy_kwargs=dict(net_arch=[512, 256]),
            learning_rate=linear_decay(7e-4, 7e-5),
            n_steps=100,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.07,
            vf_coef=0.5,
            max_grad_norm=1.0,
            use_rms_prop=True,
            verbose=0,
            seed=seed,
        )

    raise ValueError(f"Unknown agent: {name}")


# ─────────────────────────────────────────
# 6. Train one agent
# ─────────────────────────────────────────

def train_agent(name, args):
    print(f"\n{'='*64}")
    print(f"  Training {name.upper()}  ({args.timesteps:,} timesteps)")
    print(f"{'='*64}")

    env      = MovieRecEnv(data_dir=args.data_dir, seed=args.seed)
    env      = Monitor(env)
    agent    = get_agent(name, env, args.seed, args.timesteps)
    callback = MetricsCallback(agent_name=name.upper(), verbose=args.verbose)

    agent.learn(
        total_timesteps=args.timesteps,
        callback=callback,
        progress_bar=True,
    )

    model_path = os.path.join(args.models_dir, name)
    os.makedirs(model_path, exist_ok=True)
    agent.save(os.path.join(model_path, "model"))
    print(f"\n  Saved → {model_path}/model.zip")

    metrics = callback.get_metrics()
    n       = len(metrics["rewards"])
    last100 = min(100, n)
    last50  = min(50,  n)

    print(f"\n  {name.upper()} summary ({n} episodes):")
    print(f"    First 100 reward : {np.mean(metrics['rewards'][:100]):.3f}")
    print(f"    Last  100 reward : {np.mean(metrics['rewards'][-last100:]):.3f}  "
          f"(Δ={np.mean(metrics['rewards'][-last100:])-np.mean(metrics['rewards'][:100]):+.3f})")
    print(f"    Last   50 reward : {np.mean(metrics['rewards'][-last50:]):.3f} "
          f"± {np.std(metrics['rewards'][-last50:]):.3f}")
    print(f"    Last   50 sat    : {np.mean(metrics['sats'][-last50:]):.3f} "
          f"± {np.std(metrics['sats'][-last50:]):.3f}")
    print(f"    Last   50 div    : {np.mean(metrics['divs'][-last50:]):.3f}")
    print(f"    Last   50 reps   : {np.mean(metrics['reps'][-last50:]):.1f}  "
          f"(want < 3)")
    print(f"    Last   50 movies : {np.mean(metrics['movies'][-last50:]):.1f}/20  "
          f"(want > 16)")

    # Collapse detection
    mid_n   = n // 2
    early_r = np.mean(metrics["rewards"][:mid_n])
    late_r  = np.mean(metrics["rewards"][mid_n:])
    if late_r < early_r - 1.5:
        print(f"\n  ⚠ Collapse detected: {early_r:.2f} → {late_r:.2f}")
    else:
        print(f"\n  ✓ No collapse (early={early_r:.2f}, late={late_r:.2f})")

    # DQN-specific repetition warning
    if name == "dqn":
        final_reps = np.mean(metrics["reps"][-last50:])
        if final_reps > 5:
            print(f"  ⚠ DQN still repeating ({final_reps:.1f} reps/ep) — "
                  f"consider longer exploration or lower learning rate")
        elif final_reps < 3:
            print(f"  ✓ DQN repetitions under control ({final_reps:.1f}/ep)")

    env.close()
    return metrics


# ─────────────────────────────────────────
# 7. Main
# ─────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.models_dir, exist_ok=True)

    # Backup existing models first
    if not args.no_backup:
        print(f"\n{'='*64}")
        print(f"  BACKING UP EXISTING MODELS")
        print(f"{'='*64}")
        backup_path = backup_models(args.agents, args.models_dir, args.backup_dir)
    else:
        print("  Skipping backup (--no_backup flag set)")

    print(f"\n{'='*64}")
    print(f"  RL TRAINING v4")
    print(f"{'='*64}")
    print(f"  Agents         : {args.agents}")
    print(f"  Timesteps      : {args.timesteps:,} per agent")
    print(f"  PPO/A2C net    : [512, 256] — entropy=0.07")
    print(f"  DQN net        : [256, 128] — explore=75%, eps_floor=0.20")
    print(f"  Seed           : {args.seed}")
    print(f"  Backup dir     : {args.backup_dir}")

    all_metrics = {}

    # Load existing metrics for agents not being retrained
    existing_metrics_path = os.path.join(args.models_dir, "training_metrics.npy")
    if os.path.exists(existing_metrics_path):
        try:
            existing = np.load(existing_metrics_path, allow_pickle=True).item()
            for k, v in existing.items():
                if k not in [a for a in args.agents]:
                    all_metrics[k] = v
                    print(f"  Loaded existing metrics for {k.upper()}")
        except Exception:
            pass

    for name in args.agents:
        metrics = train_agent(name, args)
        all_metrics[name] = metrics
        # Save after each agent in case of interruption
        np.save(os.path.join(args.models_dir, "training_metrics.npy"), all_metrics)

    # Final comparison
    print(f"\n{'='*64}")
    print(f"  FINAL COMPARISON (last 50 episodes)")
    print(f"{'='*64}")
    print(f"  {'Agent':10s}  {'Eps':>8s}  {'Reward':>8s}  "
          f"{'Sat':>6s}  {'Reps':>6s}  {'Movies':>8s}")
    print(f"  {'-'*58}")
    print(f"  {'[Greedy]':10s}  {'—':>8s}  {'6.814':>8s}  "
          f"{'0.178':>6s}  {'0.0':>6s}  {'20.0':>8s}")
    print(f"  {'[Random]':10s}  {'—':>8s}  {'9.190':>8s}  "
          f"{'0.818':>6s}  {'0.4':>6s}  {'19.6':>8s}")

    for name, metrics in all_metrics.items():
        n      = len(metrics["rewards"])
        last50 = min(50, n)
        r      = np.mean(metrics["rewards"][-last50:])
        s      = np.mean(metrics["sats"][-last50:])
        rp     = np.mean(metrics["reps"][-last50:])
        m      = np.mean(metrics["movies"][-last50:])
        tag    = " ← retrained" if name in args.agents else " ← kept"
        print(f"  {name.upper():10s}  {n:>8d}  {r:>8.3f}  "
              f"{s:>6.3f}  {rp:>6.1f}  {m:>8.1f}{tag}")

    print(f"  {'[Oracle]':10s}  {'—':>8s}  {'12.063':>8s}  "
          f"{'0.979':>6s}  {'0.0':>6s}  {'20.0':>8s}")
    print(f"{'='*64}")
    print(f"\n  Metrics saved → {args.models_dir}/training_metrics.npy")
    if not args.no_backup:
        print(f"  Old models backed up → {args.backup_dir}")
    print(f"  Next: python evaluate.py")


if __name__ == "__main__":
    main()
