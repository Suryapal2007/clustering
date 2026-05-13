import asyncio
import sys

# Fix for WinError 10054: An existing connection was forcibly closed by the remote host
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px

# Setting the page config for that "aesthetic" look
st.set_page_config(page_title="TrendCluster AI", page_icon="✨", layout="wide")

st.title("✨ TrendCluster AI: What's the Vibe?")
st.markdown("### *Gen-Z Edition: Automatically grouping the internet's brainrot.*")

# 1. Generating a Random "Gen-Z" Dataset
@st.cache_data
def get_data():
    data = {
        "post": [
            "skibidi toilet is actually deep lore", "rizz god in the building", 
            "how to get more aura points", "fanum tax is getting out of hand",
            "this new sustainable fashion drop is mid", "thrifting haul for the summer",
            "is fast fashion cooked?", "capsule wardrobe for minimalist vibes",
            "bitcoin hitting a new high lets go", "is ethereum a buy right now?",
            "crypto wallets for beginners", "passive income via defi",
            "best skincare for glowing skin", "night routine for clear skin",
            "is slugging still a thing in 2024?", "morning matcha and glass skin"
        ]
    }
    return pd.DataFrame(data)

df = get_data()

# Sidebar for "The Controls"
st.sidebar.header("🎛️ Customize the Vibes")
num_clusters = st.sidebar.slider("How many clusters (groups)?", 2, 5, 3)
theme_color = st.sidebar.selectbox("Pick a Chart Aesthetic", ["Viridis", "Prism", "Plotly_dark"])

# 2. The ML Logic (Vectorization + Clustering)
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['post'])

model = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
df['cluster'] = model.fit_predict(X)

# 3. Dimensionality Reduction for Visualization (PCA)
# We turn high-dim text data into 2D so we can actually see it
pca = PCA(n_components=2)
coords = pca.fit_transform(X.toarray())
df['x'] = coords[:, 0]
df['y'] = coords[:, 1]

# 4. The Interactive Dashboard
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 Cluster Visualization")
    fig = px.scatter(df, x='x', y='y', color='cluster', 
                 hover_data=['post'], 
                 title="Posts grouped by 'Similarity'",
                 color_continuous_scale=theme_color)
    fig.update_traces(marker=dict(size=15, line=dict(width=2, color='DarkSlateGrey')))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("🔍 Raw Data & Groups")
    st.dataframe(df[['cluster', 'post']].sort_values(by='cluster'), use_container_width=True)

# 5. Explaining the "Aura" of the Clusters
st.divider()
st.subheader("💡 What just happened?")
st.info(f"We took raw text, turned it into numbers (math is the real rizz), and grouped them into {num_clusters} distinct 'vibes' using K-Means Clustering.")