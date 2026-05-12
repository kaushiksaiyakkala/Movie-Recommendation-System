# ============================================================
# utils/tmdb_api.py
# USING OMDb INSTEAD
# ============================================================

import requests
import streamlit as st

# ============================================================
# API KEY
# ============================================================

OMDB_API_KEY = "ae538910"

# ============================================================
# FETCH POSTER
# ============================================================

@st.cache_data(show_spinner=False)
def get_movie_poster(movie_title):

    try:

        clean_title = movie_title.split("(")[0].strip()

        url = (
            f"http://www.omdbapi.com/"
            f"?apikey={OMDB_API_KEY}"
            f"&t={clean_title}"
        )

        response = requests.get(url)

        data = response.json()

        poster = data.get("Poster")

        if poster is None:
            return None

        if poster == "N/A":
            return None

        return poster

    except:
        return None
        return None