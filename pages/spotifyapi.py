import base64
import os

import pandas as pd
import streamlit as st
from sklearn.neighbors import NearestNeighbors
try:
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials
except ImportError:
    spotipy = None

from config import DATA_PATH, IMAGE_PATH


st.set_page_config(page_title="Spotify API", page_icon="🎧", layout="wide")


def add_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"<style>.stApp {{background-image: url('data:image/jpg;base64,{encoded}');"
        "background-size: cover; background-position: center;}}</style>",
        unsafe_allow_html=True,
    )


if IMAGE_PATH.exists():
    add_bg_from_local(IMAGE_PATH)

client_id = os.getenv("SPOTIFY_CLIENT_ID")
client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
sp = None
if spotipy and client_id and client_secret:
    sp = spotipy.Spotify(
        auth_manager=SpotifyClientCredentials(
            client_id=client_id, client_secret=client_secret
        )
    )


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


df = load_data()
features = [
    column
    for column in [
        "danceability",
        "energy",
        "valence",
        "tempo",
        "acousticness",
        "liveness",
        "instrumentalness",
        "loudness",
    ]
    if column in df.columns
]
model = NearestNeighbors(n_neighbors=min(6, len(df)), metric="cosine").fit(
    df[features]
)


def get_song_features(song_name, artists=None):
    if sp is None:
        return None
    query = song_name + (f" artist:{artists}" if artists else "")
    result = sp.search(q=query, type="track", limit=1)
    items = result["tracks"]["items"]
    if not items:
        return None
    track = items[0]
    try:
        audio = sp.audio_features([track["id"]])[0]
        values = [audio[column] for column in features]
    except (TypeError, KeyError, IndexError):
        values = [0] * len(features)
    return {
        "track_name": track["name"],
        "artists": track["artists"][0]["name"],
        "id": track["id"],
        "preview_url": track.get("preview_url"),
        "features": values,
    }


def recommend_from_dataset(song_features, n=5):
    _, indices = model.kneighbors([song_features])
    return df.iloc[indices[0][1 : n + 1]][["track_name", "artists"]]


st.title("🎵 Spotify API Recommendation System")
if sp is None:
    st.info("Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET to enable API search.")

song_name = st.text_input("Enter a song name")
artists = st.text_input("Enter artist name (optional)")

if st.button("Recommend"):
    song_data = get_song_features(song_name, artists)
    if not song_data:
        st.error("Spotify API credentials are missing or the song was not found.")
    else:
        st.success(f"🎵 Found: {song_data['track_name']} by {song_data['artists']}")
        if song_data["preview_url"]:
            st.audio(song_data["preview_url"])
        st.markdown(
            f"https://open.spotify.com/embed/track/{song_data['id']}",
            unsafe_allow_html=False,
        )
        st.dataframe(recommend_from_dataset(song_data["features"]))
