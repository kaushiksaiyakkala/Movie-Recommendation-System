# ============================================================
# app.py
# RL-BASED MOVIE RECOMMENDER SYSTEM
# STREAMLIT DEMO
# ============================================================

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import pandas as pd
import numpy as np
import random
from utils.tmdb_api import get_movie_poster

from utils.inference import MovieRecommender

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="RL Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_resource
def load_recommender():

    return MovieRecommender()

@st.cache_data
def load_movies():

    return pd.read_csv("data/movies.csv")

# ------------------------------------------------------------

movies_df = load_movies()

recommender = load_recommender()

# ============================================================
# SESSION STATE
# ============================================================

if "initialized" not in st.session_state:

    st.session_state.initialized = False

if "history" not in st.session_state:

    st.session_state.history = []

if "recommendations" not in st.session_state:

    st.session_state.recommendations = []

# ============================================================
# HEADER
# ============================================================

st.title("🎬 RL-Based Adaptive Movie Recommender")

st.markdown("""
This system uses:

- Neural Collaborative Filtering
- GRU-based User State Encoding
- Reinforcement Learning Re-ranking
- PPO / A2C / DQN comparison
- FAISS Candidate Retrieval

to generate adaptive long-term movie recommendations.
""")

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Settings")

model_choice = st.sidebar.selectbox(
    "RL Model",
    ["PPO", "A2C", "DQN"]
)

exploration = st.sidebar.slider(
    "Exploration vs Safe Recommendations",
    0.0,
    1.0,
    0.3
)

num_recommendations = st.sidebar.slider(
    "Number of Recommendations",
    5,
    20,
    10
)

# ============================================================
# USER INITIALIZATION
# ============================================================

if not st.session_state.initialized:

    st.subheader("🎥 Select Movies You Like")

    movie_titles = movies_df["title"].tolist()

    selected_movies = st.multiselect(
        "Choose at least 5 movies",
        movie_titles
    )

    if st.button("Initialize Recommender"):

        if len(selected_movies) < 5:

            st.warning(
                "Please select at least 5 movies."
            )

        else:

            recommender.initialize_user(
                selected_movies
            )

            st.session_state.history = selected_movies

            st.session_state.initialized = True

            st.success(
                "User profile initialized!"
            )

            st.rerun()

# ============================================================
# MAIN RECOMMENDATION UI
# ============================================================

else:

    st.subheader("🍿 Recommended Movies")

    recommendations = recommender.recommend(
        top_k=num_recommendations,
        model_name=model_choice.lower(),
        exploration_weight=exploration
    )

    st.session_state.recommendations = recommendations

    # --------------------------------------------------------
    # DISPLAY MOVIES
    # --------------------------------------------------------

    cols = st.columns(2)

    for idx, rec in enumerate(recommendations):

        col = cols[idx % 2]

        with col:

            st.markdown("---")

            poster_url = get_movie_poster(
                rec["title"]
            )

            if poster_url is not None:

                st.image(
                    poster_url,
                    width=220
                )

            st.subheader(rec["title"])

            st.write(
                f"🎭 Genres: {rec['genres']}"
            )

            st.write(
                f"⭐ Predicted Reward: "
                f"{rec['reward']:.3f}"
            )

            st.write(
                f"🌍 Diversity Score: "
                f"{rec['diversity']:.3f}"
            )

            # ------------------------------------------------
            # FEEDBACK BUTTONS
            # ------------------------------------------------

            like_col, dislike_col = st.columns(2)

            with like_col:

                if st.button(
                    f"👍 Like {idx}"
                ):

                    recommender.update_feedback(
                        rec["movie_idx"],
                        reward=1.0
                    )

                    st.success(
                        f"Liked {rec['title']}"
                    )

                    st.rerun()

            with dislike_col:

                if st.button(
                    f"👎 Dislike {idx}"
                ):

                    recommender.update_feedback(
                        rec["movie_idx"],
                        reward=-1.0
                    )

                    st.warning(
                        f"Disliked {rec['title']}"
                    )

                    st.rerun()

    # ========================================================
    # USER HISTORY
    # ========================================================

    st.markdown("---")

    st.subheader("🕘 Recent Interaction History")

    history = recommender.get_recent_history()

    for movie in history[-10:]:

        st.write(f"• {movie}")

    # ========================================================
    # ANALYTICS
    # ========================================================

    st.markdown("---")

    st.subheader("📊 Session Analytics")

    analytics = recommender.get_analytics()

    metric1, metric2, metric3 = st.columns(3)

    with metric1:

        st.metric(
            "Total Recommendations",
            analytics["total_recommendations"]
        )

    with metric2:

        st.metric(
            "Average Reward",
            f"{analytics['average_reward']:.3f}"
        )

    with metric3:

        st.metric(
            "Diversity Score",
            f"{analytics['diversity_score']:.3f}"
        )

    # ========================================================
    # RESET
    # ========================================================

    st.markdown("---")

    if st.button("🔄 Reset Session"):

        st.session_state.initialized = False

        st.session_state.history = []

        st.session_state.recommendations = []

        recommender.reset()

        st.rerun()