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

    streamlit run app.py                        # digits by default
    streamlit run app.py -- --dataset emnist-letters   # letters pre-selected
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

from model import VAE

# Artefact files per dataset. MNIST keeps legacy names; EMNIST-letters
# (A-Z alphabet) gets its own files so both can coexist.
DATASET_ASSETS = {
    "MNIST digits (0-9)": {
        "weights": "vae_weights.pth",
        "csv": "latent_data.csv",
        "bounds": "latent_bounds.json",
        "meta": "dataset_mnist.json",
        "keywords": ("mnist", "digit", "digits", "0-9", "numbers"),
    },
    "EMNIST letters (A-Z)": {
        "weights": "vae_emnist_letters_weights.pth",
        "csv": "latent_emnist_letters.csv",
        "bounds": "latent_emnist_letters_bounds.json",
        "meta": "dataset_emnist_letters.json",
        "keywords": ("emnist", "emnist-letters", "letter", "letters", "alphabet", "a-z"),
    },
}


def default_dataset(requested: str | None, available: list[str]) -> int:
    """Index into ``available`` matching a ``--dataset`` CLI keyword.

    Lets launcher scripts (e.g. ``runalphabets.bat`` passes
    ``-- --dataset emnist-letters``) pre-select a dataset. Unknown or
    missing values fall back to index 0 — never crash the dashboard.
    """
    if requested:
        want = requested.strip().lower()
        for i, name in enumerate(available):
            if want in DATASET_ASSETS[name]["keywords"]:
                return i
    return 0


def requested_dataset_arg() -> str | None:
    """Read ``--dataset <name>`` from the script args (after ``--``)."""
    args = sys.argv[1:]
    # Streamlit consumes its own flags; ours come after a `--` separator.
    if "--" in args:
        args = args[args.index("--") + 1 :]
    for i, a in enumerate(args):
        if a == "--dataset" and i + 1 < len(args):
            return args[i + 1]
        if a.startswith("--dataset="):
            return a.split("=", 1)[1]
    return None


def _palette(n: int):
    """Distinct colors for n classes (10 -> tab10, more -> hsv)."""
    if n <= 10:
        return plt.get_cmap("tab10").colors
    cmap = plt.get_cmap("hsv")
    return [cmap(i / n) for i in range(n)]


# ---------------------------------------------------------------------------
# Cached loading: Streamlit re-runs the script on every interaction, so we
# cache heavy operations (model loading, CSV parsing) with @st.cache_resource.
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model(weights_path: str) -> VAE:
    """Load trained VAE weights onto CPU and set eval mode."""
    model = VAE(z_dim=2)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"'{weights_path}' not found. Run `python train.py` first."
        )
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


@st.cache_data
def load_data(csv_path: str, bounds_path: str):
    """Load the latent-space CSV and the optional precomputed bounds."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"'{csv_path}' not found. Run `python train.py` first."
        )
    df = pd.read_csv(csv_path)
    # Normalise columns: new CSVs use label/label_name, old ones use digit.
    if "label" not in df.columns and "digit" in df.columns:
        df["label"] = df["digit"]
    if "label_name" not in df.columns:
        df["label_name"] = df["label"].astype(str)
    if os.path.exists(bounds_path):
        bounds = json.load(open(bounds_path))
    else:
        bounds = {
            "x_min": float(df.latent_x.min()) - 0.1,
            "x_max": float(df.latent_x.max()) + 0.1,
            "y_min": float(df.latent_y.min()) - 0.1,
            "y_max": float(df.latent_y.max()) + 0.1,
        }
    return df, bounds


def draw_latent_scatter(df: pd.DataFrame, zoom: tuple[float, float, float, float]):
    """Render the 2D latent space coloured by class (digit or letter)."""
    classes = sorted(df["label"].unique().tolist())
    colors = _palette(len(classes))
    color_of = {c: colors[i] for i, c in enumerate(classes)}
    name_of = df.drop_duplicates("label").set_index("label")["label_name"].to_dict()
    fig, ax = plt.subplots(figsize=(6, 5), dpi=110)
    for c in classes:
        subset = df[df.label == c]
        ax.scatter(
            subset.latent_x,
            subset.latent_y,
            s=8,
            alpha=0.6,
            label=str(name_of.get(c, c)),
            color=color_of[c],
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
    "Train a 2D-latent VAE (`python train.py --dataset mnist` or "
    "`--dataset emnist-letters`), then drag the sliders "
    "to morph between characters decoded from the latent space."
)

available = {k: v for k, v in DATASET_ASSETS.items() if os.path.exists(v["weights"]) and os.path.exists(v["csv"])}
if not available:
    st.error(
        "No trained models found. Run `python train.py --dataset mnist` or "
        "`python train.py --dataset emnist-letters --epochs 3 --max-train 20000` first."
    )
    st.stop()

# Visible dataset picker at the top of the page (not hidden in the
# sidebar) so nobody misses it. `runalphabets.bat` pre-selects letters
# by passing `-- --dataset emnist-letters`.
options = list(available.keys())
preselect = default_dataset(requested_dataset_arg(), options)
choice = st.radio("Dataset — pick what the AI draws", options, index=preselect, horizontal=True)
assets = available[choice]

try:
    model = load_model(assets["weights"])
    df, bounds = load_data(assets["csv"], assets["bounds"])
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