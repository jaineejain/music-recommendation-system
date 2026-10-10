import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from config import DATA_PATH

st.set_page_config(page_title="Explore Dataset", page_icon="📊", layout="wide")

st.title("📊 Explore the Dataset")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

st.write("### Dataset Preview")
st.dataframe(df.head(100))

st.write("### Top 10 Artists by Song Count")
top_artists = df["artists"].value_counts().head(10)
fig, ax = plt.subplots()
top_artists.plot(kind="bar", ax=ax)
st.pyplot(fig)

st.write("### Feature Distributions")
feature = st.selectbox("Choose a Feature:", ["danceability", "energy", "tempo", "loudness"])
fig, ax = plt.subplots()
df[feature].hist(ax=ax, bins=30)
st.pyplot(fig)
