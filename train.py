"""
train.py
========
Train the Convolutional VAE on MNIST and export the artefacts the
Streamlit dashboard needs:

  * ``vae_weights.pth``  - trained model state dict
  * ``latent_data.csv``  - 2D latent coordinates + digit label for 1000 test images

The script runs on whatever device is available (CUDA > MPS > CPU), so
it can be launched simply with::

    python train.py
"""

from __future__ import annotations

import json
import os
import random

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import VAE, vae_loss

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 42
EPOCHS = 5
BATCH_SIZE = 128
LR = 1e-3
Z_DIM = 2
N_LATENT_SAMPLES = 1000  # number of test images to project into the latent space

WEIGHTS_PATH = "vae_weights.pth"
LATENT_PATH = "latent_data.csv"


def set_seed(seed: int) -> None:
    """Reproducibility for CPU, GPU and DataLoader workers."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    """Automatically pick CUDA, then MPS (Apple Silicon), then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_mnist():
    """Download/prepare MNIST once and return (train_loader, test_dataset)."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # 0-1 float tensors shaped 1x28x28
        ]
    )
    train_data = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_data = datasets.MNIST("./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    return train_loader, test_data


def train_vae(
    model: VAE,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = EPOCHS,
) -> None:
    """Run the training loop, printing a per-epoch loss summary."""
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    model.train()

    for epoch in range(1, epochs + 1):
        running_total = 0.0
        running_recon = 0.0
        running_kl = 0.0
        n_batches = 0

        for x, _ in train_loader:
            x = x.to(device)
            recon, mu, logvar = model(x)

            total, recon_loss, kl_loss = vae_loss(recon, x, mu, logvar)

            optimizer.zero_grad()
            total.backward()
            optimizer.step()

            running_total += total.item()
            running_recon += recon_loss.item()
            running_kl += kl_loss.item()
            n_batches += 1

        avg_total = running_total / n_batches
        avg_recon = running_recon / n_batches
        avg_kl = running_kl / n_batches
        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"loss={avg_total:.2f} recon={avg_recon:.2f} kl={avg_kl:.2f}"
        )


@torch.no_grad()
def build_latent_csv(
    model: VAE,
    test_data,
    device: torch.device,
    n_samples: int = N_LATENT_SAMPLES,
) -> None:
    """Encode N test images, save (x, y, label) rows to ``latent_data.csv``."""
    model.eval()

    # Deterministic subsample for reproducibility.
    loader = DataLoader(test_data, batch_size=256, shuffle=False)
    zs, labels = [], []
    count = 0
    for x, y in loader:
        if count >= n_samples:
            break
        x = x.to(device)
        mu, _ = model.encoder(x)  # use the latent mean as the representative point
        zs.append(mu.cpu())
        labels.append(y)
        count += x.size(0)

    z = torch.cat(zs, dim=0)[:n_samples]
    y = torch.cat(labels, dim=0)[:n_samples]

    df = pd.DataFrame(
        {
            "latent_x": z[:, 0].numpy(),
            "latent_y": z[:, 1].numpy(),
            "digit": y.numpy().astype(int),
        }
    )
    df.to_csv(LATENT_PATH, index=False)
    print(f"Saved latent coordinates for {len(df)} images -> {LATENT_PATH}")
    print(f"Latent ranges: x [{df.latent_x.min():.2f}, {df.latent_x.max():.2f}], "
          f"y [{df.latent_y.min():.2f}, {df.latent_y.max():.2f}]")


def main() -> None:
    set_seed(SEED)
    device = get_device()
    print(f"Using device: {device}")

    train_loader, test_data = load_mnist()

    model = VAE(z_dim=Z_DIM).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    train_vae(model, train_loader, device, epochs=EPOCHS)

    # ---- Save trained weights -------------------------------------------
    torch.save(model.state_dict(), WEIGHTS_PATH)
    print(f"Saved model weights -> {WEIGHTS_PATH}")

    # ---- Extract and save the 2D latent coordinates ----------------------
    build_latent_csv(model, test_data, device)

    # ---- Also dump a small sidecar describing the latent-space bounds so
    # ---- the dashboard can align its sliders with the scatter plot. -----
    df = pd.read_csv(LATENT_PATH)
    bounds = {
        "x_min": float(df.latent_x.min()),
        "x_max": float(df.latent_x.max()),
        "y_min": float(df.latent_y.min()),
        "y_max": float(df.latent_y.max()),
    }
    with open("latent_bounds.json", "w") as fh:
        json.dump(bounds, fh, indent=2)
    print("Saved latent bounds -> latent_bounds.json")


if __name__ == "__main__":
    main()