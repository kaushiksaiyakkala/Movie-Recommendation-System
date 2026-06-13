# 🎬 RL-Based Adaptive Movie Recommendation System

An adaptive movie recommendation engine that combines **Neural Collaborative Filtering**, **GRU-based sequential user state encoding**, and **Reinforcement Learning re-ranking** (PPO, A2C, DQN) to generate personalized, long-term movie recommendations — trained on the MovieLens 32M dataset.

## Overview

Traditional recommenders optimize for immediate relevance, often leading to repetitive or narrow suggestions. This project frames recommendation as a **sequential decision-making problem**: an RL agent learns a policy that balances relevance, diversity, novelty, and long-term user engagement.

### Pipeline

1. **Preprocessing** – Clean MovieLens 32M ratings/movies data, filter to top movies and active users, build chronological user interaction sequences.
2. **Embeddings** – Generate movie embeddings for content representation.
3. **State Space (GRU Encoder)** – A GRU-based encoder learns a latent "user state" from sequential viewing history, capturing evolving taste and preference drift.
4. **Reward Model** – A neural reward model approximates user satisfaction, relevance, diversity, and novelty for a given (state, movie) pair.
5. **Candidate Generation (FAISS)** – A FAISS index retrieves the top-K candidate movies for a given user state, so the RL agent only reranks a small candidate set instead of the full catalog.
6. **RL Re-ranking** – PPO, A2C, and DQN agents are trained to re-rank candidates for optimal long-term reward.
7. **Evaluation & Demo** – Compare policies (Greedy vs PPO vs A2C vs DQN) on metrics like reward, diversity, repetition, and engagement, with plots and a Streamlit demo.

## Project Structure

```
.
├── app.py                      # Main Streamlit demo app
├── requirements.txt
├── data/                        # Processed data, embeddings, FAISS index
│   ├── movies.csv
│   ├── movie_embeddings_normalized.npy
│   ├── movie_id_map.pkl
│   └── faiss_movie_index.bin
├── models/                      # Trained model weights
│   ├── gru_state_encoder.pth
│   ├── reward_model.pth
│   ├── ppo_movie_recommender.zip
│   ├── a2c_movie_recommender.zip
│   └── dqn_movie_recommender.zip
├── training code/                # Training pipeline scripts
│   ├── preprocess.py             # Stage 1-2: Data preprocessing
│   ├── embeddings.py              # Movie embedding generation
│   ├── statespace.py              # Stage 3: GRU user state encoder
│   ├── reward.py                  # Stage 4: Neural reward model
│   ├── cadidate generation.py     # Stage 5: FAISS candidate retrieval
│   ├── ppo re-ranking.py           # RL training (PPO/A2C/DQN)
│   ├── evaluation.py               # Policy evaluation
│   └── generate_plots.py           # Results visualization
├── utils/
│   ├── inference.py               # Recommendation inference logic
│   ├── tmdb_api.py                 # TMDB poster fetching
│   └── utils.py
├── Frontend/
│   ├── app.py                     # Technical demo Streamlit app
│   └── demo_episode.py             # Fixed-episode policy comparison demo
└── RLfinalprojectreport (5).pdf   # Project report
```

## Setup

```bash
git clone https://github.com/kaushiksaiyakkala/Movie-Recommendation-System.git
cd Movie-Recommendation-System
pip install -r requirements.txt
```

### Requirements

- Python 3.10+
- streamlit
- torch, torchvision, torchaudio
- numpy, pandas, scikit-learn, scipy
- matplotlib
- stable-baselines3
- gymnasium
- faiss-cpu
- tqdm, pillow, requests

## Usage

### Run the main demo app

```bash
streamlit run app.py
```

This launches an interactive UI where you can:
- Choose an RL policy (PPO / A2C / DQN)
- Adjust the exploration vs. safe-recommendation slider
- Set the number of recommendations
- View movie posters fetched via the TMDB API

### Run the technical demo (policy comparison)

```bash
cd Frontend
streamlit run app.py
```

Or run the standalone episode comparison script:

```bash
python demo_episode.py
```

This runs a fixed 20-step recommendation episode using Greedy, PPO, A2C, and DQN policies on the same user history, then reports reward, diversity, repetition, and engagement metrics, saving CSVs/plots to `demo_outputs/`.

## Training Pipeline

To retrain from scratch, run the scripts in `training code/` in order:

1. `preprocess.py` – Build processed datasets from raw MovieLens ratings/movies CSVs (update the hardcoded `RATINGS_PATH` / `MOVIES_PATH` to your local MovieLens 32M files).
2. `embeddings.py` – Generate movie embeddings.
3. `statespace.py` – Train the GRU user state encoder.
4. `reward.py` – Train the neural reward model.
5. `cadidate generation.py` – Build the FAISS candidate index.
6. `ppo re-ranking.py` – Train PPO/A2C/DQN re-ranking agents.
7. `evaluation.py` / `generate_plots.py` – Evaluate trained policies and generate result plots.

## Models

| Component | Architecture | File |
|---|---|---|
| User State Encoder | GRU (state dim 128) | `models/gru_state_encoder.pth` |
| Reward Model | Neural network | `models/reward_model.pth` |
| Candidate Retrieval | FAISS index | `data/faiss_movie_index.bin` |
| Re-ranking Policy | PPO | `models/ppo_movie_recommender.zip` |
| Re-ranking Policy | A2C | `models/a2c_movie_recommender.zip` |
| Re-ranking Policy | DQN | `models/dqn_movie_recommender.zip` |

## Dataset

Built on the [MovieLens 32M dataset](https://grouplens.org/datasets/movielens/32m/), filtered to the top 5,000 most-interacted movies and users with at least 20 interactions.

## Report

See `RLfinalprojectreport (5).pdf` for the full project writeup, methodology, and results.

## License

No license specified.
