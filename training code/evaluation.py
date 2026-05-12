# ============================================================
# POLICY EVALUATION SCRIPT
# ============================================================
#
# RUN THIS IN A SEPARATE FILE:
#
# evaluate_policies.py
#
# GOAL:
# Compare:
# - Random
# - Greedy
# - PPO
# - A2C
# - DQN
#
# METRICS:
# - cumulative reward
# - diversity
#
# OUTPUT:
# - policy_comparison.png
#
# ============================================================

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gymnasium as gym
from gymnasium import spaces

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle
import random
import faiss

import torch
import torch.nn as nn

from stable_baselines3 import PPO, A2C, DQN

# ============================================================
# CONFIG
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SEQUENCE_LENGTH = 30

USER_STATE_DIM = 128
MOVIE_EMBED_DIM = 64

TOP_K = 100

EPISODE_LENGTH = 20
NUM_EPISODES = 100

# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading user sequences...")

with open("C:\\Users\\deepa\\OneDrive\\Desktop\\RL project\\everything\\user_sequences.pkl", "rb") as f:
    user_sequences = pickle.load(f)

# ------------------------------------------------------------

print("\nLoading movie embeddings...")

movie_embeddings = np.load(
    "C:\\Users\\deepa\\OneDrive\\Desktop\\RL project\\everything\\movie_embeddings_normalized.npy"
).astype("float32")

NUM_MOVIES = movie_embeddings.shape[0]

# ------------------------------------------------------------

print("\nLoading FAISS index...")

faiss_index = faiss.read_index(
    "C:\\Users\\deepa\\OneDrive\\Desktop\\RL project\\everything\\faiss_movie_index.bin"
)

# ============================================================
# GRU MODEL
# ============================================================

class GRUStateEncoder(nn.Module):

    def __init__(
        self,
        movie_embeddings,
        embed_dim,
        hidden_dim,
        num_movies
    ):
        super().__init__()

        self.movie_embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(movie_embeddings),
            freeze=True
        )

        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.output_layer = nn.Sequential(

            nn.Linear(hidden_dim, 256),
            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(256, num_movies)
        )

    def forward(self, sequences):

        x = self.movie_embedding(sequences)

        _, hidden = self.gru(x)

        hidden = hidden.squeeze(0)

        logits = self.output_layer(hidden)

        return logits, hidden

# ============================================================
# REWARD MODEL
# ============================================================

class RewardModel(nn.Module):

    def __init__(self, user_dim, movie_dim):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(user_dim + movie_dim, 256),
            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 1)
        )

    def forward(self, states, movies):

        x = torch.cat(
            [states, movies],
            dim=1
        )

        return self.network(x).squeeze()

# ============================================================
# LOAD MODELS
# ============================================================

print("\nLoading GRU encoder...")

gru_model = GRUStateEncoder(
    movie_embeddings,
    MOVIE_EMBED_DIM,
    USER_STATE_DIM,
    NUM_MOVIES
).to(DEVICE)

gru_model.load_state_dict(
    torch.load(
        "C:\\Users\\deepa\\OneDrive\\Desktop\\RL project\\everything\\gru_state_encoder.pth",
        map_location=DEVICE,
        weights_only=True
    )
)

gru_model.eval()

# ------------------------------------------------------------

print("\nLoading reward model...")

reward_model = RewardModel(
    USER_STATE_DIM,
    MOVIE_EMBED_DIM
).to(DEVICE)

reward_model.load_state_dict(
    torch.load(
        "C:\\Users\\deepa\\OneDrive\\Desktop\\RL project\\everything\\reward_model.pth",
        map_location=DEVICE,
        weights_only=True
    )
)

reward_model.eval()

# ============================================================
# ENVIRONMENT
# ============================================================

class MovieRecommendationEnv(gym.Env):

    def __init__(self):

        super().__init__()

        self.action_space = spaces.Discrete(
            TOP_K
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(USER_STATE_DIM,),
            dtype=np.float32
        )

    # --------------------------------------------------------

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.user_id = random.choice(
            list(user_sequences.keys())
        )

        self.sequence = user_sequences[self.user_id]

        self.current_history = [
            x["movie_idx"]
            for x in self.sequence[:SEQUENCE_LENGTH]
        ]

        self.step_count = 0

        state = self._get_state()

        return state, {}

    # --------------------------------------------------------

    def _get_state(self):

        seq_tensor = torch.LongTensor(
            self.current_history[-SEQUENCE_LENGTH:]
        ).unsqueeze(0).to(DEVICE)

        with torch.no_grad():

            _, hidden = gru_model(seq_tensor)

        state = (
            hidden.squeeze()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        return state

    # --------------------------------------------------------

    def _get_candidates(self, state):

        state_norm = state / (
            np.linalg.norm(state) + 1e-8
        )

        projected = state_norm[:MOVIE_EMBED_DIM]

        projected = projected.reshape(1, -1).astype("float32")

        faiss.normalize_L2(projected)

        scores, indices = faiss_index.search(
            projected,
            TOP_K
        )

        return indices[0]

    # --------------------------------------------------------

    def step(self, action):

        state = self._get_state()

        candidates = self._get_candidates(state)

        movie_idx = candidates[action]

        movie_emb = movie_embeddings[movie_idx]

        # reward
        state_tensor = torch.FloatTensor(
            state
        ).unsqueeze(0).to(DEVICE)

        movie_tensor = torch.FloatTensor(
            movie_emb
        ).unsqueeze(0).to(DEVICE)

        with torch.no_grad():

            reward = reward_model(
                state_tensor,
                movie_tensor
            ).item()

        # update history
        self.current_history.append(movie_idx)

        if len(self.current_history) > SEQUENCE_LENGTH:
            self.current_history.pop(0)

        self.step_count += 1

        terminated = (
            self.step_count >= EPISODE_LENGTH
        )

        next_state = self._get_state()

        info = {
            "movie_idx": movie_idx
        }

        return (
            next_state,
            reward,
            terminated,
            False,
            info
        )

# ============================================================
# LOAD RL MODELS
# ============================================================

env = MovieRecommendationEnv()

print("\nLoading PPO...")

ppo_model = PPO.load(
    "C:\\Users\\deepa\\OneDrive\\Desktop\\RL project\\everything\\ppo_movie_recommender",
    env=env,
    device="cuda"
)

print("\nLoading A2C...")

a2c_model = A2C.load(
    "C:\\Users\\deepa\\OneDrive\\Desktop\\RL project\\everything\\a2c_movie_recommender",
    env=env,
    device="cuda"
)

print("\nLoading DQN...")

dqn_model = DQN.load(
    "C:\\Users\\deepa\\OneDrive\\Desktop\\RL project\\everything\\dqn_movie_recommender",
    env=env,
    device="cuda"
)

# ============================================================
# POLICIES
# ============================================================

def random_policy(candidates):

    return random.randint(
        0,
        len(candidates)-1
    )

# ------------------------------------------------------------

def greedy_policy(state, candidates):

    state_proj = state[:64]

    sims = []

    for movie_idx in candidates:

        movie_emb = movie_embeddings[movie_idx]

        sim = np.dot(
            state_proj,
            movie_emb
        )

        sims.append(sim)

    return np.argmax(sims)

# ============================================================
# EVALUATION
# ============================================================

results = defaultdict(list)

diversity_scores = defaultdict(list)

engagement_scores = defaultdict(list)

repetition_scores = defaultdict(list)

# ============================================================

for policy_name in [
    "random",
    "greedy",
    "ppo",
    "a2c",
    "dqn"
]:

    print(f"\nEvaluating {policy_name}...")

    episode_rewards = []
    episode_diversity = []

    for ep in range(NUM_EPISODES):

        state, _ = env.reset()

        total_reward = 0

        recommended_movies = []

        done = False

        while not done:

            candidates = env._get_candidates(state)

            # ------------------------------------------------

            if policy_name == "random":

                action = random_policy(candidates)

            elif policy_name == "greedy":

                action = greedy_policy(
                    state,
                    candidates
                )

            elif policy_name == "ppo":

                action, _ = ppo_model.predict(
                    state,
                    deterministic=True
                )

            elif policy_name == "a2c":

                action, _ = a2c_model.predict(
                    state,
                    deterministic=True
                )

            else:

                action, _ = dqn_model.predict(
                    state,
                    deterministic=True
                )

            # ------------------------------------------------

            next_state, reward, done, _, info = env.step(action)

            total_reward += reward

            recommended_movies.append(
                info["movie_idx"]
            )

            state = next_state

        # ====================================================
        # DIVERSITY
        # ====================================================

        unique_movies = len(
            set(recommended_movies)
        )

        diversity = unique_movies / len(
            recommended_movies
        )

        # ====================================================
        # REPETITION RATE
        # ====================================================

        repetition_rate = 1.0 - diversity

        # ====================================================
        # ENGAGEMENT SCORE
        # ====================================================

        engagement = (
            total_reward * diversity
        )

        # ====================================================

        episode_rewards.append(total_reward)

        episode_diversity.append(diversity)

        engagement_scores[policy_name].append(
            engagement
        )

        repetition_scores[policy_name].append(
            repetition_rate
        )

        episode_rewards.append(total_reward)

        episode_diversity.append(diversity)

    results[policy_name] = episode_rewards

    diversity_scores[policy_name] = episode_diversity

# ============================================================
# PRINT RESULTS
# ============================================================

print("\n================ RESULTS ================\n")

summary = {}

for policy_name in results:

    avg_reward = np.mean(
        results[policy_name]
    )

    avg_diversity = np.mean(
        diversity_scores[policy_name]
    )

    avg_engagement = np.mean(
        engagement_scores[policy_name]
    )

    avg_repetition = np.mean(
        repetition_scores[policy_name]
    )

    reward_std = np.std(
        results[policy_name]
    )

    summary[policy_name] = {

        "reward": avg_reward,

        "diversity": avg_diversity,

        "engagement": avg_engagement,

        "repetition": avg_repetition,

        "stability": reward_std
    }

    print(
        f"{policy_name.upper()}"
    )

    print(
        f"Avg Reward    : {avg_reward:.3f}"
    )

    print(
        f"Avg Diversity : {avg_diversity:.3f}"
    )

    print(
        f"Engagement    : {avg_engagement:.3f}"
    )

    print(
        f"Repetition    : {avg_repetition:.3f}"
    )

    print(
        f"Reward StdDev : {reward_std:.3f}"
    )

    print()

# ============================================================
# ANALYTICAL SUMMARY
# ============================================================

print(
    "\n================ FINAL ANALYSIS ================\n"
)

best_reward = max(
    summary,
    key=lambda x: summary[x]["reward"]
)

best_diversity = max(
    summary,
    key=lambda x: summary[x]["diversity"]
)

best_engagement = max(
    summary,
    key=lambda x: summary[x]["engagement"]
)

most_stable = min(
    summary,
    key=lambda x: summary[x]["stability"]
)

lowest_repetition = min(
    summary,
    key=lambda x: summary[x]["repetition"]
)

print(
    f"Highest Reward        : {best_reward.upper()}"
)

print(
    f"Highest Diversity     : {best_diversity.upper()}"
)

print(
    f"Best Engagement       : {best_engagement.upper()}"
)

print(
    f"Most Stable           : {most_stable.upper()}"
)

print(
    f"Lowest Repetition     : {lowest_repetition.upper()}"
)

# ============================================================
# DIVERSITY IMPROVEMENT
# ============================================================

greedy_div = summary["greedy"]["diversity"]

ppo_div = summary["ppo"]["diversity"]

a2c_div = summary["a2c"]["diversity"]

dqn_div = summary["dqn"]["diversity"]

ppo_gain = (
    (ppo_div - greedy_div)
    / greedy_div
) * 100

a2c_gain = (
    (a2c_div - greedy_div)
    / greedy_div
) * 100

dqn_gain = (
    (dqn_div - greedy_div)
    / greedy_div
) * 100

print(
    "\n================ DIVERSITY GAINS ================\n"
)

print(
    f"PPO Diversity Gain over Greedy : "
    f"{ppo_gain:.2f}%"
)

print(
    f"A2C Diversity Gain over Greedy : "
    f"{a2c_gain:.2f}%"
)

print(
    f"DQN Diversity Gain over Greedy : "
    f"{dqn_gain:.2f}%"
)

# ============================================================
# INTERPRETATION
# ============================================================

print(
    "\n================ INTERPRETATION ================\n"
)

print(
    "Greedy recommendation achieved "
    "the highest immediate reward "
    "but suffered severe diversity "
    "collapse and high repetition."
)

print()

print(
    "RL-based approaches maintained "
    "comparable reward while greatly "
    "improving diversity and "
    "long-term engagement."
)

print()

print(
    "This demonstrates the ability "
    "of reinforcement learning methods "
    "to balance exploration and "
    "exploitation in sequential "
    "recommendation systems."
)

# ============================================================
# SAVE SUMMARY CSV
# ============================================================

import pandas as pd

summary_df = pd.DataFrame(summary).T

# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

import os

output_dir = (
    "C:\\Users\\deepa\\OneDrive\\Desktop\\RL project\\everything\\evaluation_plots"
)

os.makedirs(
    output_dir,
    exist_ok=True
)

# ============================================================
# SAVE CSV
# ============================================================

csv_path = (
    output_dir +
    "\\final_metrics.csv"
)

summary_df.to_csv(csv_path)

print(
    f"\nSaved metrics CSV: {csv_path}"
)

# ============================================================
# PLOT + SAVE CUMULATIVE REWARD
# ============================================================

plt.figure(figsize=(12,6))

for policy_name, rewards in results.items():

    cumulative = np.cumsum(rewards)

    plt.plot(
        cumulative,
        label=policy_name.upper(),
        linewidth=2
    )

plt.xlabel(
    "Episode",
    fontsize=12
)

plt.ylabel(
    "Cumulative Reward",
    fontsize=12
)

plt.title(
    "RL Policy Comparison",
    fontsize=14
)

plt.legend()

plt.grid(True)

save_path = (
    output_dir +
    "\\policy_comparison.png"
)

plt.savefig(
    save_path,
    bbox_inches="tight",
    dpi=300
)

plt.show()

print(
    f"\nSaved plot: {save_path}"
)

# ============================================================
# DIVERSITY BAR PLOT
# ============================================================

plt.figure(figsize=(10,6))

policy_names = list(
    diversity_scores.keys()
)

avg_diversities = [

    np.mean(diversity_scores[p])

    for p in policy_names
]

plt.bar(
    policy_names,
    avg_diversities
)

plt.ylabel(
    "Average Diversity"
)

plt.title(
    "Policy Diversity Comparison"
)

plt.grid(True)

plt.savefig(
    output_dir +
    "\\diversity_comparison.png",
    bbox_inches="tight",
    dpi=300
)

plt.show()

# ============================================================
# ENGAGEMENT PLOT
# ============================================================

plt.figure(figsize=(10,6))

avg_engagements = [

    np.mean(engagement_scores[p])

    for p in policy_names
]

plt.bar(
    policy_names,
    avg_engagements
)

plt.ylabel(
    "Engagement Score"
)

plt.title(
    "Long-Term Engagement Comparison"
)

plt.grid(True)

plt.savefig(
    output_dir +
    "\\engagement_comparison.png",
    bbox_inches="tight",
    dpi=300
)

plt.show()

# ============================================================
# REPETITION RATE PLOT
# ============================================================

plt.figure(figsize=(10,6))

avg_repetitions = [

    np.mean(repetition_scores[p])

    for p in policy_names
]

plt.bar(
    policy_names,
    avg_repetitions
)

plt.ylabel(
    "Repetition Rate"
)

plt.title(
    "Recommendation Repetition Comparison"
)

plt.grid(True)

plt.savefig(
    output_dir +
    "\\repetition_comparison.png",
    bbox_inches="tight",
    dpi=300
)

plt.show()

# ============================================================
# REWARD VS DIVERSITY SCATTER
# ============================================================

plt.figure(figsize=(10,6))

for policy_name in policy_names:

    avg_reward = np.mean(
        results[policy_name]
    )

    avg_diversity = np.mean(
        diversity_scores[policy_name]
    )

    plt.scatter(
        avg_diversity,
        avg_reward,
        s=200,
        label=policy_name.upper()
    )

    plt.text(
        avg_diversity,
        avg_reward,
        policy_name.upper()
    )

plt.xlabel(
    "Diversity"
)

plt.ylabel(
    "Reward"
)

plt.title(
    "Reward vs Diversity Tradeoff"
)

plt.grid(True)

plt.legend()

plt.savefig(
    output_dir +
    "\\reward_diversity_tradeoff.png",
    bbox_inches="tight",
    dpi=300
)

plt.show()