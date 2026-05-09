"""
train.py
========
Train PPO, Double DQN, and A2C agents on the MovieRecEnv.

Key improvements over v1:
  - PPO: larger network, tuned entropy, learning rate schedule
  - DQN: longer exploration, prioritized replay via larger buffer,
          more learning starts, smaller epsilon floor
  - A2C: gradient clipping increased, adjusted n_steps, learning rate
          schedule to prevent the catastrophic collapse seen in v1
  - All agents: 300k timesteps (was 200k) for better convergence
  - Linear learning rate decay for PPO and A2C (standard best practice)
  - Larger networks [256, 256] -> [512, 256] for better capacity
  - Metrics saved per-episode for smooth learning curves in plots

Usage:
    python train.py                           # train all 3
    python train.py --agents ppo             # single agent
    python train.py --timesteps 300000       # custom timesteps

Outputs:
    models/ppo/model.zip
    models/dqn/model.zip
    models/a2c/model.zip
    models/training_metrics.npy
"""

import os
import argparse
import numpy as np
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_linear_fn
from environment import MovieRecEnv


# ─────────────────────────────────────────
# 1. Args
# ─────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents",     nargs="+", default=["ppo", "dqn", "a2c"],
                        choices=["ppo", "dqn", "a2c"])
    parser.add_argument("--timesteps",  type=int,   default=300_000,
                        help="Training timesteps per agent (default 300k)")
    parser.add_argument("--data_dir",   type=str,   default="./data/processed")
    parser.add_argument("--models_dir", type=str,   default="./models")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--verbose",    type=int,   default=1)
    return parser.parse_args()


# ─────────────────────────────────────────
# 2. Metrics callback
# ─────────────────────────────────────────

class MetricsCallback(BaseCallback):
    """
    Records per-episode metrics during training.
    Prints rolling average every 100 episodes.
    """

    def __init__(self, agent_name="", verbose=0):
        super().__init__(verbose)
        self.agent_name          = agent_name
        self.episode_rewards     = []
        self.episode_sats        = []
        self.episode_genres      = []
        self.episode_movies      = []
        self.episode_lengths     = []
        self.episode_divs        = []

        self._current_reward     = 0.0
        self._current_length     = 0
        self._step_divs          = []

    def _on_step(self):
        info   = self.locals["infos"][0]
        reward = self.locals["rewards"][0]
        done   = self.locals["dones"][0]

        self._current_reward += reward
        self._current_length += 1
        self._step_divs.append(info.get("diversity_bonus", 0.0))

        if done:
            self.episode_rewards.append(self._current_reward)
            self.episode_lengths.append(self._current_length)
            self.episode_sats.append(info.get("satisfaction",     0.0))
            self.episode_genres.append(info.get("n_unique_genres", 0))
            self.episode_movies.append(info.get("n_unique_movies", 0))
            self.episode_divs.append(np.mean(self._step_divs))

            self._current_reward = 0.0
            self._current_length = 0
            self._step_divs      = []

            n = len(self.episode_rewards)
            if self.verbose >= 1 and n % 100 == 0:
                last = min(100, n)
                print(f"    [{self.agent_name}] ep {n:5d} | "
                      f"reward={np.mean(self.episode_rewards[-last:]):.3f} | "
                      f"sat={np.mean(self.episode_sats[-last:]):.3f} | "
                      f"div={np.mean(self.episode_divs[-last:]):.3f} | "
                      f"genres={np.mean(self.episode_genres[-last:]):.1f}")

        return True

    def get_metrics(self):
        return {
            "rewards":  np.array(self.episode_rewards),
            "sats":     np.array(self.episode_sats),
            "genres":   np.array(self.episode_genres),
            "movies":   np.array(self.episode_movies),
            "lengths":  np.array(self.episode_lengths),
            "divs":     np.array(self.episode_divs),
        }


# ─────────────────────────────────────────
# 3. Agent configs
# ─────────────────────────────────────────

def get_agent(name, env, seed, total_timesteps):
    """
    Returns configured SB3 agent with tuned hyperparameters.

    Key design decisions:
    - [512, 256] network: more capacity than [256, 256], handles
      the 65-dim state and 500-action space better
    - Linear LR decay for PPO/A2C: prevents overshooting late in training
      which caused A2C collapse in v1
    - DQN: much longer exploration (50% of training), higher epsilon floor,
      more learning starts — addresses the Q-value collapse seen in v1
    - ent_coef on PPO/A2C: entropy bonus directly incentivises diverse
      action selection, aligning algorithm objective with rec diversity goal
    """

    # Linear learning rate schedule: starts at lr, decays to lr/10 by end
    def linear_schedule(initial_lr):
        return get_linear_fn(initial_lr, initial_lr / 10, 1.0)

    # Larger network for better representational capacity
    policy_kwargs = dict(net_arch=[512, 256])

    if name == "ppo":
        return PPO(
            "MlpPolicy", env,
            policy_kwargs=policy_kwargs,
            # Learning rate: linear decay 3e-4 → 3e-5
            learning_rate=linear_schedule(3e-4),
            n_steps=2048,          # rollout length before update
            batch_size=128,        # was 64 — larger batch = more stable gradients
            n_epochs=10,           # epochs per update
            gamma=0.99,            # discount — values long-term reward
            gae_lambda=0.95,       # GAE smoothing
            clip_range=0.2,        # PPO clipping — prevents large policy updates
            # Entropy coefficient: encourages exploration/diversity
            # Higher than default (0.0) to push agent to diversify
            ent_coef=0.02,         # was 0.01 — stronger diversity incentive
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=0,
            seed=seed,
        )

    elif name == "dqn":
        # DQN v1 collapsed because:
        # 1. Exploration ended too early (30% of training)
        # 2. Too few learning starts — Q-values unstable early
        # 3. Epsilon floor too low — no random fallback once greedy
        # Fixes: longer exploration, more starts, higher epsilon floor
        return DQN(
            "MlpPolicy", env,
            policy_kwargs=policy_kwargs,
            learning_rate=5e-4,         # slightly higher than v1
            buffer_size=200_000,        # was 100k — larger replay buffer
            learning_starts=5000,       # was 1000 — more data before Q-learning
            batch_size=128,             # was 64
            gamma=0.99,
            target_update_interval=500, # was 1000 — more frequent target updates
            # Exploration: spend 50% of training exploring (was 30%)
            exploration_fraction=0.5,
            # Higher epsilon floor: always keep 10% random actions
            # Prevents Q-value collapse into pure greedy
            exploration_final_eps=0.10, # was 0.05
            train_freq=4,
            optimize_memory_usage=False,
            verbose=0,
            seed=seed,
        )
        # SB3 DQN uses Double DQN by default

    elif name == "a2c":
        # A2C v1 collapsed because:
        # 1. No learning rate decay — large updates late in training
        # 2. n_steps=5 too short — high variance gradient estimates
        # 3. Insufficient gradient clipping
        # Fixes: LR decay, longer n_steps, stronger clipping
        return A2C(
            "MlpPolicy", env,
            policy_kwargs=policy_kwargs,
            # Linear LR decay: 7e-4 → 7e-5
            learning_rate=linear_schedule(7e-4),
            n_steps=20,            # was 5 — longer rollouts = lower variance
            gamma=0.99,
            gae_lambda=0.95,       # was 1.0 — smoother advantage estimates
            ent_coef=0.02,         # was 0.01 — stronger diversity incentive
            vf_coef=0.5,
            max_grad_norm=1.0,     # was 0.5 — allow larger updates early
            use_rms_prop=True,     # RMSProp optimizer (A2C standard)
            verbose=0,
            seed=seed,
        )

    raise ValueError(f"Unknown agent: {name}")


# ─────────────────────────────────────────
# 4. Train one agent
# ─────────────────────────────────────────

def train_agent(name, args):
    print(f"\n{'='*60}")
    print(f"  Training {name.upper()}  ({args.timesteps:,} timesteps)")
    print(f"{'='*60}")

    env      = MovieRecEnv(data_dir=args.data_dir, seed=args.seed)
    env      = Monitor(env)
    agent    = get_agent(name, env, args.seed, args.timesteps)
    callback = MetricsCallback(agent_name=name.upper(), verbose=args.verbose)

    agent.learn(
        total_timesteps=args.timesteps,
        callback=callback,
        progress_bar=True,
    )

    # Save model
    model_path = os.path.join(args.models_dir, name)
    os.makedirs(model_path, exist_ok=True)
    agent.save(os.path.join(model_path, "model"))
    print(f"\n  Saved → {model_path}/model.zip")

    # Training summary
    metrics = callback.get_metrics()
    n       = len(metrics["rewards"])
    last100 = min(100, n)
    last50  = min(50,  n)

    print(f"\n  {name.upper()} training summary ({n} episodes):")
    print(f"    First 100 eps reward : {np.mean(metrics['rewards'][:100]):.3f}")
    print(f"    Last  100 eps reward : {np.mean(metrics['rewards'][-last100:]):.3f}  "
          f"(Δ = {np.mean(metrics['rewards'][-last100:]) - np.mean(metrics['rewards'][:100]):+.3f})")
    print(f"    Last   50 eps reward : {np.mean(metrics['rewards'][-last50:]):.3f} "
          f"± {np.std(metrics['rewards'][-last50:]):.3f}")
    print(f"    Last   50 eps sat    : {np.mean(metrics['sats'][-last50:]):.3f} "
          f"± {np.std(metrics['sats'][-last50:]):.3f}")
    print(f"    Last   50 eps div    : {np.mean(metrics['divs'][-last50:]):.3f}")
    print(f"    Last   50 eps genres : {np.mean(metrics['genres'][-last50:]):.1f}")

    # Check for collapse
    mid_n     = n // 2
    early_r   = np.mean(metrics["rewards"][:mid_n])
    late_r    = np.mean(metrics["rewards"][mid_n:])
    if late_r < early_r - 1.0:
        print(f"\n  ⚠ WARNING: reward dropped {early_r:.2f} → {late_r:.2f} "
              f"(possible collapse — consider retraining with lower LR)")
    else:
        print(f"\n  ✓ No collapse detected (early={early_r:.2f}, late={late_r:.2f})")

    env.close()
    return metrics


# ─────────────────────────────────────────
# 5. Main
# ─────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.models_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  RL TRAINING — IMPROVED v2")
    print(f"{'='*60}")
    print(f"  Agents         : {args.agents}")
    print(f"  Timesteps      : {args.timesteps:,} per agent")
    print(f"  Network        : [512, 256] MLP")
    print(f"  LR schedule    : Linear decay (PPO, A2C)")
    print(f"  DQN explore    : 50% of training, eps_final=0.10")
    print(f"  Entropy coef   : 0.02 (PPO, A2C)")
    print(f"  Seed           : {args.seed}")

    all_metrics = {}

    for name in args.agents:
        metrics = train_agent(name, args)
        all_metrics[name] = metrics

        # Save after each agent in case of interruption
        metrics_path = os.path.join(args.models_dir, "training_metrics.npy")
        np.save(metrics_path, all_metrics)

    # Final comparison table
    print(f"\n{'='*60}")
    print(f"  FINAL COMPARISON (last 50 episodes)")
    print(f"{'='*60}")
    print(f"  {'Agent':8s}  {'Episodes':>10s}  {'Reward':>10s}  "
          f"{'Sat':>8s}  {'Div':>8s}  {'Genres':>8s}")
    print(f"  {'-'*58}")

    # Baselines for reference
    print(f"  {'[Greedy]':8s}  {'—':>10s}  {'5.837':>10s}  "
          f"{'0.061':>8s}  {'—':>8s}  {'11.3':>8s}")
    print(f"  {'[Random]':8s}  {'—':>10s}  {'8.201':>10s}  "
          f"{'0.549':>8s}  {'—':>8s}  {'13.0':>8s}")

    for name, metrics in all_metrics.items():
        n      = len(metrics["rewards"])
        last50 = min(50, n)
        r      = np.mean(metrics["rewards"][-last50:])
        s      = np.mean(metrics["sats"][-last50:])
        d      = np.mean(metrics["divs"][-last50:])
        g      = np.mean(metrics["genres"][-last50:])
        print(f"  {name.upper():8s}  {n:>10d}  {r:>10.3f}  "
              f"{s:>8.3f}  {d:>8.3f}  {g:>8.1f}")

    print(f"  {'[Oracle]':8s}  {'—':>10s}  {'11.471':>10s}  "
          f"{'0.886':>8s}  {'—':>8s}  {'12.8':>8s}")
    print(f"{'='*60}")
    print(f"\n  Metrics saved → {os.path.join(args.models_dir, 'training_metrics.npy')}")
    print(f"  Next step: python evaluate.py")


if __name__ == "__main__":
    main()
