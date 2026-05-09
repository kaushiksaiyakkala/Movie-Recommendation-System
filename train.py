"""
train.py
========
Train PPO, Double DQN, and A2C agents on the MovieRecEnv v3.

Key changes in v3:
  - PPO: higher entropy coef (0.05) to strongly discourage repetition,
          larger batch (256), more n_steps (4096)
  - DQN: even longer exploration (60%), higher epsilon floor (0.15),
          train_freq=1 for more frequent updates
  - A2C: higher entropy (0.05), longer n_steps (50), stronger grad clip
  - All: 400k timesteps for better convergence with harder environment

Usage:
    python train.py
    python train.py --agents ppo
    python train.py --timesteps 400000
"""

import os
import argparse
import numpy as np
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from environment import MovieRecEnv


# ─────────────────────────────────────────
# 1. Args
# ─────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents",     nargs="+", default=["ppo", "dqn", "a2c"],
                        choices=["ppo", "dqn", "a2c"])
    parser.add_argument("--timesteps",  type=int,   default=400_000)
    parser.add_argument("--data_dir",   type=str,   default="./data/processed")
    parser.add_argument("--models_dir", type=str,   default="./models")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--verbose",    type=int,   default=1)
    return parser.parse_args()


# ─────────────────────────────────────────
# 2. Linear LR schedule (SB3 compatible)
# ─────────────────────────────────────────

def linear_decay(initial_lr, final_lr=1e-5):
    """Returns a callable LR schedule for SB3."""
    def schedule(progress_remaining):
        # progress_remaining: 1.0 at start → 0.0 at end
        return final_lr + progress_remaining * (initial_lr - final_lr)
    return schedule


# ─────────────────────────────────────────
# 3. Metrics callback
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
            if self.verbose >= 1 and n % 100 == 0:
                last = min(100, n)
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
# 4. Agent configs
# ─────────────────────────────────────────

def get_agent(name, env, seed, total_timesteps):
    """
    v3 hyperparameter changes:

    PPO:
      - ent_coef 0.02 → 0.05: much stronger entropy bonus to actively
        discourage the repetition strategy agents found in v2
      - n_steps 2048 → 4096: longer rollouts see full episode consequences
      - batch_size 128 → 256: larger batch for more stable updates
      - LR decay 3e-4 → 3e-5

    DQN:
      - exploration_fraction 0.5 → 0.6: even longer exploration
      - exploration_final_eps 0.10 → 0.15: higher random floor
        prevents locking into repeat strategy during exploitation
      - train_freq 4 → 1: update every step for faster learning
      - target_update_interval 500 → 250: more frequent target sync

    A2C:
      - ent_coef 0.02 → 0.05: same entropy fix as PPO
      - n_steps 20 → 50: much longer rollouts, sees full episode
      - LR decay 7e-4 → 7e-5
    """
    policy_kwargs = dict(net_arch=[512, 256])

    if name == "ppo":
        return PPO(
            "MlpPolicy", env,
            policy_kwargs=policy_kwargs,
            learning_rate=linear_decay(3e-4, 3e-5),
            n_steps=4096,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.05,         # strong entropy — discourages repetition
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=0,
            seed=seed,
        )

    elif name == "dqn":
        return DQN(
            "MlpPolicy", env,
            policy_kwargs=policy_kwargs,
            learning_rate=5e-4,
            buffer_size=200_000,
            learning_starts=5000,
            batch_size=128,
            gamma=0.99,
            target_update_interval=250,
            exploration_fraction=0.6,
            exploration_final_eps=0.15,
            train_freq=1,          # update every step
            verbose=0,
            seed=seed,
        )

    elif name == "a2c":
        return A2C(
            "MlpPolicy", env,
            policy_kwargs=policy_kwargs,
            learning_rate=linear_decay(7e-4, 7e-5),
            n_steps=50,            # sees ~2.5 full episodes per update
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.05,         # strong entropy
            vf_coef=0.5,
            max_grad_norm=1.0,
            use_rms_prop=True,
            verbose=0,
            seed=seed,
        )

    raise ValueError(f"Unknown agent: {name}")


# ─────────────────────────────────────────
# 5. Train one agent
# ─────────────────────────────────────────

def train_agent(name, args):
    print(f"\n{'='*62}")
    print(f"  Training {name.upper()}  ({args.timesteps:,} timesteps)")
    print(f"{'='*62}")

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

    # Collapse check
    mid_n   = n // 2
    early_r = np.mean(metrics["rewards"][:mid_n])
    late_r  = np.mean(metrics["rewards"][mid_n:])
    if late_r < early_r - 1.5:
        print(f"\n  ⚠ Possible collapse: {early_r:.2f} → {late_r:.2f}")
    else:
        print(f"\n  ✓ No collapse (early={early_r:.2f}, late={late_r:.2f})")

    env.close()
    return metrics


# ─────────────────────────────────────────
# 6. Main
# ─────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.models_dir, exist_ok=True)

    print(f"\n{'='*62}")
    print(f"  RL TRAINING v3")
    print(f"{'='*62}")
    print(f"  Agents         : {args.agents}")
    print(f"  Timesteps      : {args.timesteps:,} per agent")
    print(f"  Network        : [512, 256] MLP")
    print(f"  Entropy coef   : 0.05 (PPO, A2C) — strong diversity incentive")
    print(f"  DQN explore    : 60% of training, eps_final=0.15")
    print(f"  Rep penalty    : scaled [0, 0.5, 0.8, 1.0]")
    print(f"  Seed           : {args.seed}")

    all_metrics = {}

    for name in args.agents:
        metrics = train_agent(name, args)
        all_metrics[name] = metrics
        # Save after each agent
        np.save(os.path.join(args.models_dir, "training_metrics.npy"), all_metrics)

    # Final table
    print(f"\n{'='*62}")
    print(f"  FINAL COMPARISON (last 50 episodes)")
    print(f"{'='*62}")
    print(f"  {'Agent':10s}  {'Eps':>8s}  {'Reward':>8s}  "
          f"{'Sat':>6s}  {'Reps':>6s}  {'Movies':>8s}")
    print(f"  {'-'*58}")
    print(f"  {'[Greedy]':10s}  {'—':>8s}  {'5.790':>8s}  "
          f"{'0.064':>6s}  {'0.0':>6s}  {'20.0':>8s}")
    print(f"  {'[Random]':10s}  {'—':>8s}  {'8.391':>8s}  "
          f"{'0.580':>6s}  {'0.4':>6s}  {'19.6':>8s}")

    for name, metrics in all_metrics.items():
        n      = len(metrics["rewards"])
        last50 = min(50, n)
        r      = np.mean(metrics["rewards"][-last50:])
        s      = np.mean(metrics["sats"][-last50:])
        rp     = np.mean(metrics["reps"][-last50:])
        m      = np.mean(metrics["movies"][-last50:])
        print(f"  {name.upper():10s}  {n:>8d}  {r:>8.3f}  "
              f"{s:>6.3f}  {rp:>6.1f}  {m:>8.1f}")

    print(f"  {'[Oracle]':10s}  {'—':>8s}  {'11.471':>8s}  "
          f"{'0.886':>6s}  {'0.0':>6s}  {'20.0':>8s}")
    print(f"{'='*62}")
    print(f"\n  Metrics saved → models/training_metrics.npy")
    print(f"  Next: python evaluate.py")


if __name__ == "__main__":
    main()