import base64
import os

import pandas as pd
import requests
import streamlit as st
from sklearn.neighbors import NearestNeighbors

from config import DATA_PATH, IMAGE_PATH


st.set_page_config(page_title="Spotify Recommender", page_icon="🎶", layout="wide")


def add_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


if IMAGE_PATH.exists():
    add_bg_from_local(IMAGE_PATH)


CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def build_knn(features):
    knn = NearestNeighbors(n_neighbors=min(10, len(features)), metric="cosine")
    knn.fit(features)
    return knn


df = load_data()
feature_cols = [
    "danceability",
    "energy",
    "valence",
    "tempo",
    "acousticness",
    "liveness",
    "instrumentalness",
    "loudness",
]
feature_cols = [column for column in feature_cols if column in df.columns]
X = df[feature_cols].values
knn = build_knn(X)


@st.cache_resource
def get_token():
    if not CLIENT_ID or not CLIENT_SECRET:
        return None
    response = requests.post(
        "https://accounts.spotify.com/api/token",
        data={
            "grant_type": "client_credentials",
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
        },
        timeout=15,
    )
    response.raise_for_status()
    return response.json().get("access_token")


def get_spotify_url(song, artist=""):
    token = get_token()
    if not token:
        return None
    query = f"track:{song}"
    if artist:
        query += f" artist:{artist}"
    response = requests.get(
        "https://api.spotify.com/v1/search",
        params={"q": query, "type": "track", "limit": 1},
        headers={"Authorization": f"Bearer {token}"},
        timeout=15,
    )
    response.raise_for_status()
    tracks = response.json().get("tracks", {}).get("items", [])
    return f"https://open.spotify.com/track/{tracks[0]['id']}" if tracks else None


df["track_name"] = df["track_name"].astype(str).str.strip().str.lower()
df["artists"] = df["artists"].astype(str).str.strip().str.lower()


def find_track_index(query_track, query_artist=None):
    query = query_track.strip().lower()
    if query_artist:
        mask = (df["track_name"] == query) & (
            df["artists"] == query_artist.strip().lower()
        )
    else:
        mask = df["track_name"] == query
    matches = df[mask]
    if matches.empty:
        matches = df[df["track_name"].str.contains(query, regex=False)]
    return matches.index[0] if not matches.empty else None


def recommend_songs(track_name, artist=None, k=10):
    index = find_track_index(track_name, artist)
    if index is None:
        return pd.DataFrame()
    count = min(k + 1, len(df))
    _, indices = knn.kneighbors(X[index].reshape(1, -1), n_neighbors=count)
    recommendation_indices = [i for i in indices.flatten() if i != index][:k]
    return df.iloc[recommendation_indices][["track_name", "artists"]]


st.title("🎶 Spotify Recommendation System")
st.write("Find similar songs based on audio features.")

song = st.text_input("Enter a Song Name:", "Blinding Lights")
artist = st.text_input("(Optional) Enter Artist Name:")

demo_spotify_links = {
    "blinding lights": "https://open.spotify.com/track/0VjIjW4GlUZAMYd2vXMi3b",
    "shape of you": "https://open.spotify.com/track/7qiZfU4dY1lWllzX7mPBI3",
    "someone you loved": "https://open.spotify.com/track/7qEHsqek33rTcFNT9PFqLf",
    "drivers license": "https://open.spotify.com/track/5wANPM4fQCJwkGd4rN57mH",
}

if st.button("Recommend"):
    recs = recommend_songs(song, artist)
    if recs.empty:
        st.warning("❌ Song not found in dataset")
    else:
        st.success("✅ Here are your recommendations:")
        for _, row in recs.reset_index(drop=True).iterrows():
            track = row["track_name"].title()
            artist_name = row["artists"].title()
            st.markdown(f"🎵 **{track}** by *{artist_name}*")
            spotify_url = get_spotify_url(track, artist_name)
            st.markdown(
                spotify_url or demo_spotify_links.get(track.lower(), ""),
                unsafe_allow_html=False,
            )
