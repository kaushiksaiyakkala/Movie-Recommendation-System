import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import pandas as pd
import streamlit as st

from demo_episode import TechnicalMovieDemo, EPISODE_LENGTH

st.set_page_config(
    page_title="RL Movie Recommendation Demo",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main { background: #0b1020; }
    [data-testid="stAppViewContainer"] { background: linear-gradient(135deg, #0b1020 0%, #111827 45%, #1f2937 100%); }
    [data-testid="stHeader"] { background: rgba(0,0,0,0); }
    [data-testid="stSidebar"] { background: #0f172a; }
    h1, h2, h3, p, label, span, div { color: #f8fafc; }
    .hero {
        padding: 1.5rem 1.75rem;
        border-radius: 1.25rem;
        background: linear-gradient(135deg, rgba(99,102,241,0.35), rgba(14,165,233,0.18));
        border: 1px solid rgba(255,255,255,0.14);
        box-shadow: 0 20px 60px rgba(0,0,0,0.35);
        margin-bottom: 1.2rem;
    }
    .hero-title {
        font-size: 2.15rem;
        font-weight: 800;
        margin-bottom: 0.25rem;
    }
    .hero-subtitle {
        color: #cbd5e1;
        font-size: 1.05rem;
        line-height: 1.55;
        max-width: 980px;
    }
    .card {
        padding: 1rem;
        border-radius: 1rem;
        background: rgba(15, 23, 42, 0.78);
        border: 1px solid rgba(255,255,255,0.12);
        box-shadow: 0 12px 40px rgba(0,0,0,0.24);
    }
    .policy-chip {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 999px;
        background: rgba(59,130,246,0.18);
        border: 1px solid rgba(96,165,250,0.32);
        color: #bfdbfe;
        font-size: 0.85rem;
        font-weight: 700;
        margin-right: 0.25rem;
    }
    .small-muted { color: #94a3b8; font-size: 0.9rem; }
    div[data-testid="stMetric"] {
        background: rgba(15,23,42,0.78);
        border: 1px solid rgba(255,255,255,0.12);
        padding: 1rem;
        border-radius: 1rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .stDataFrame { border-radius: 1rem; overflow: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource(show_spinner="Loading GRU, reward model, FAISS index, PPO, A2C, and DQN...")
def load_demo_engine():
    return TechnicalMovieDemo()

def find_project_root(start: Path) -> Path:
    """Find the nearest folder at or above this file that contains data/ and models/."""
    for folder in [start, *start.parents]:
        if (folder / "data").exists() and (folder / "models").exists():
            return folder
    raise FileNotFoundError(
        "Could not find data/ and models/. Put app.py and demo_episode.py in your "
        "Movie-Recommendation-System folder, or copy data/ and models/ next to them."
    )

@st.cache_data
def load_movie_titles():
    project_root = find_project_root(Path(__file__).resolve().parent)
    data_path = project_root / "data" / "movies.csv"
    df = pd.read_csv(data_path)
    df["label"] = df["title"] + "  |  " + df["genres"].fillna("")
    return df


def format_metric(value, ndigits=3):
    try:
        return f"{float(value):.{ndigits}f}"
    except Exception:
        return str(value)


def show_step_cards(step_df):
    cols = st.columns(4)
    policy_order = ["GREEDY", "PPO", "A2C", "DQN"]
    for col, policy in zip(cols, policy_order):
        row = step_df[step_df["policy"] == policy]
        with col:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"<span class='policy-chip'>{policy}</span>", unsafe_allow_html=True)
            if row.empty:
                st.write("No result")
            else:
                r = row.iloc[0]
                st.markdown(f"### {r['movie']}")
                st.markdown(f"<p class='small-muted'>{r['genres']}</p>", unsafe_allow_html=True)
                st.metric("Reward this step", format_metric(r["reward"]))
                st.metric("Cumulative reward", format_metric(r["cumulative_reward"]))
                st.metric("Diversity so far", format_metric(r["diversity_so_far"]))
            st.markdown("</div>", unsafe_allow_html=True)


st.markdown(
    """
    <div class="hero">
        <div class="hero-title">RL Movie Recommendation Demo</div>
        <div class="hero-subtitle">
            Pick a starting movie history, then run one 20-step recommendation episode for Greedy, PPO, A2C, and DQN. 
            The demo shows the actual movie selected at every timestep and tracks reward, diversity, repetition, and engagement.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

movies_df = load_movie_titles()
label_to_title = dict(zip(movies_df["label"], movies_df["title"]))
all_labels = movies_df["label"].tolist()

def find_label(title_contains):
    matches = movies_df[movies_df["title"].str.contains(title_contains, case=False, regex=False, na=False)]
    if matches.empty:
        return None
    return matches.iloc[0]["label"]

default_titles = ["Toy Story", "Finding Nemo", "Shrek", "Monsters, Inc.", "Incredibles, The"]
default_labels = [x for x in [find_label(t) for t in default_titles] if x is not None]

with st.sidebar:
    st.header("Demo setup")
    selected_labels = st.multiselect(
        "Choose starting liked movies",
        all_labels,
        default=default_labels,
        help="These movies simulate the user's recent watch history.",
    )
    episode_len_text = f"Episode length: {EPISODE_LENGTH} timesteps"
    st.caption(episode_len_text)
    run_button = st.button("Run recommendation episode", type="primary", use_container_width=True)
    st.markdown("---")

if len(selected_labels) < 5:
    st.warning("Pick at least 5 movies to initialize the user history.")
    st.stop()

selected_titles = [label_to_title[x] for x in selected_labels]

st.subheader("Starting user history")
start_cols = st.columns(min(5, len(selected_titles)))
for i, title in enumerate(selected_titles[:10]):
    with start_cols[i % len(start_cols)]:
        genre = movies_df.loc[movies_df["title"] == title, "genres"].iloc[0]
        st.markdown(
            f"<div class='card'><b>{title}</b><br><span class='small-muted'>{genre}</span></div>",
            unsafe_allow_html=True,
        )

if not run_button:
    st.info("Click **Run recommendation episode** in the sidebar.")
    st.stop()

try:
    demo = load_demo_engine()
    with st.spinner("Running Greedy, PPO, A2C, and DQN on the same starting history..."):
        episode_df, summary_df = demo.run_all(selected_titles)
except Exception as e:
    st.error("The demo crashed while loading/running the models.")
    st.exception(e)
    st.stop()

policy_order = ["GREEDY", "PPO", "A2C", "DQN"]
summary_df["policy"] = pd.Categorical(summary_df["policy"], categories=policy_order, ordered=True)
summary_df = summary_df.sort_values("policy")

st.subheader("Final metrics")
metric_cols = st.columns(4)
for col, policy in zip(metric_cols, policy_order):
    row = summary_df[summary_df["policy"] == policy]
    with col:
        if not row.empty:
            r = row.iloc[0]
            st.metric(f"{policy} engagement", format_metric(r["final_engagement"]))
            st.caption(
                f"Reward {format_metric(r['final_reward'])} | Diversity {format_metric(r['final_diversity'])} | Repetition {format_metric(r['final_repetition'])}"
            )

step = st.slider("Inspect timestep", 1, EPISODE_LENGTH, 1)
step_df = episode_df[episode_df["timestep"] == step].copy()
st.subheader(f"Timestep {step}: recommendation from each policy")
show_step_cards(step_df)

st.subheader("Metrics over time")
tab1, tab2, tab3, tab4 = st.tabs(["Cumulative reward", "Diversity", "Repetition", "Engagement"])
with tab1:
    chart_df = episode_df.pivot(index="timestep", columns="policy", values="cumulative_reward")
    st.line_chart(chart_df)
with tab2:
    chart_df = episode_df.pivot(index="timestep", columns="policy", values="diversity_so_far")
    st.line_chart(chart_df)
with tab3:
    chart_df = episode_df.pivot(index="timestep", columns="policy", values="repetition_so_far")
    st.line_chart(chart_df)
with tab4:
    chart_df = episode_df.pivot(index="timestep", columns="policy", values="engagement_so_far")
    st.line_chart(chart_df)

st.subheader("Full 20-step recommendation table")
pivot = episode_df.pivot(index="timestep", columns="policy", values="movie").reset_index()
st.dataframe(pivot, use_container_width=True, hide_index=True)

with st.expander("Show detailed per-step metrics"):
    display_df = episode_df.copy()
    for c in ["reward", "cumulative_reward", "diversity_so_far", "repetition_so_far", "engagement_so_far"]:
        display_df[c] = display_df[c].map(lambda x: round(float(x), 3))
    st.dataframe(display_df, use_container_width=True, hide_index=True)

st.download_button(
    "Download episode recommendations CSV",
    episode_df.to_csv(index=False).encode("utf-8"),
    file_name="episode_recommendations.csv",
    mime="text/csv",
)
st.download_button(
    "Download final metrics CSV",
    summary_df.to_csv(index=False).encode("utf-8"),
    file_name="final_metrics.csv",
    mime="text/csv",
)
