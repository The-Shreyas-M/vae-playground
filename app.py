"""
app.py
======
Interactive VAE Latent Space Explorer — Streamlit dashboard.

Loads the trained weights (``vae_weights.pth``) and the latent projections
(``latent_data.csv``), then lets the user explore the 2D latent space:

    * Left column : 2D scatter of the latent space, coloured by digit.
    * Right column: two sliders (latent-x, latent-y) bounded by the data.
    * Below       : the decoder output for the current (x, y) point,
                    rendered in real time as an image.

Run with::

    streamlit run app.py
"""

from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

from model import VAE

WEIGHTS_PATH = "vae_weights.pth"
LATENT_PATH = "latent_data.csv"
BOUNDS_PATH = "latent_bounds.json"

DIGIT_COLORS = plt.get_cmap("tab10").colors  # 10 colours, one per digit


# ---------------------------------------------------------------------------
# Cached loading: Streamlit re-runs the script on every interaction, so we
# cache heavy operations (model loading, CSV parsing) with @st.cache_resource.
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model() -> VAE:
    """Load the trained VAE weights onto CPU and set eval mode."""
    model = VAE(z_dim=2)
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(
            f"'{WEIGHTS_PATH}' not found. Run `python train.py` first."
        )
    state = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


@st.cache_data
def load_data():
    """Load the latent-space CSV and the optional precomputed bounds."""
    if not os.path.exists(LATENT_PATH):
        raise FileNotFoundError(
            f"'{LATENT_PATH}' not found. Run `python train.py` first."
        )
    df = pd.read_csv(LATENT_PATH)
    if os.path.exists(BOUNDS_PATH):
        bounds = json.load(open(BOUNDS_PATH))
    else:
        bounds = {
            "x_min": float(df.latent_x.min()) - 0.1,
            "x_max": float(df.latent_x.max()) + 0.1,
            "y_min": float(df.latent_y.min()) - 0.1,
            "y_max": float(df.latent_y.max()) + 0.1,
        }
    return df, bounds


def draw_latent_scatter(df: pd.DataFrame, zoom: tuple[float, float, float, float]):
    """Render the 2D latent space coloured by digit class."""
    fig, ax = plt.subplots(figsize=(6, 5), dpi=110)
    for digit in range(10):
        subset = df[df.digit == digit]
        ax.scatter(
            subset.latent_x,
            subset.latent_y,
            s=8,
            alpha=0.6,
            label=str(digit),
            color=DIGIT_COLORS[digit],
        )
    ax.set_xlim(zoom[0], zoom[1])
    ax.set_ylim(zoom[2], zoom[3])
    ax.set_xlabel("Latent x")
    ax.set_ylabel("Latent y")
    ax.set_title("Latent space (test set projections)")
    ax.legend(markerscale=3, loc="upper right", ncol=2)
    ax.grid(alpha=0.3)
    return fig


def decode_point(model: VAE, x: float, y: float) -> np.ndarray:
    """Run the (x, y) latent point through the decoder; returns a 28x28 image."""
    z = torch.tensor([[x, y]], dtype=torch.float32)
    with torch.no_grad():
        img = model.decode(z)  # shape (1, 1, 28, 28), values in [0, 1]
    return img.squeeze().numpy()


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
st.set_page_config(page_title="VAE Latent Space Explorer", layout="wide")
st.title("Interactive VAE Latent Space Explorer")
st.caption(
    "Train a 2D-latent VAE on MNIST (`python train.py`), then drag the sliders "
    "to morph between digits decoded from the latent space."
)

try:
    model = load_model()
    df, bounds = load_data()
except FileNotFoundError as err:
    st.error(err)
    st.stop()

x_min, x_max, y_min, y_max = bounds["x_min"], bounds["x_max"], bounds["y_min"], bounds["y_max"]

# Sidebar controls for the plot itself.
with st.sidebar:
    st.header("Plot controls")
    show_labels = st.checkbox("Show point markers on decoded plot", False)
    sample_size = st.slider("Points to display", 100, len(df), min(1000, len(df)), step=100)
    df_view = df.sample(n=sample_size, random_state=42) if sample_size < len(df) else df

zoom = (x_min, x_max, y_min, y_max)

left, right = st.columns([1.6, 1.0])

with left:
    st.pyplot(draw_latent_scatter(df_view, zoom))

with right:
    st.header("Latent coordinates")
    lat_x = st.slider("Latent x", min_value=x_min, max_value=x_max, value=0.0, step=0.05, format="%.2f")
    lat_y = st.slider("Latent y", min_value=y_min, max_value=y_max, value=0.0, step=0.05, format="%.2f")

    st.subheader("Decoded image")
    img = decode_point(model, lat_x, lat_y)
    st.image(img, width=200, caption=f"z = ({lat_x:.2f}, {lat_y:.2f})", clamp=True)

    # Optionally mark the current point on the scatter plot above.
    st.divider()
    st.markdown(f"**Current point:** `x = {lat_x:.2f}`, `y = {lat_y:.2f}`")