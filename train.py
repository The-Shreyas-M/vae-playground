"""
train.py
========
Train the Convolutional VAE on MNIST (digits) or an alphabet dataset
(EMNIST Letters A-Z) and export the artefacts the Streamlit dashboard needs:

  * ``<weights>`` - trained model state dict
  * ``<latent_csv>`` - 2D latent coordinates + label for test images
  * ``<bounds>`` - min/max of the latent space (for slider alignment)
  * ``<meta>`` - dataset name + class names (so the app can label the plot)

The script runs on whatever device is available (CUDA > MPS > CPU), so
it can be launched simply with::

    python train.py --dataset mnist
    python train.py --dataset emnist-letters --epochs 3 --max-train 20000

EMNIST Letters note: torchvision labels are 1-26 (1=A). We convert to
0-25 internally and store the letter (A-Z) in the CSV for the app.
EMNIST images also come out transposed vs MNIST, so we transpose them
back (see ``emnist_transform``) — otherwise letters appear rotated.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import string

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
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

DATASETS = ("mnist", "emnist-letters", "fashion-mnist", "kmnist")

# Default artefact filenames per dataset. MNIST keeps the legacy names so
# `streamlit run app.py` works unchanged; other datasets get their own
# files so both can coexist and the app can switch between them.
DATASET_FILES = {
    "mnist": {
        "weights": "vae_weights.pth",
        "csv": "latent_data.csv",
        "bounds": "latent_bounds.json",
        "meta": "dataset_mnist.json",
        "label": "digit",
    },
    "emnist-letters": {
        "weights": "vae_emnist_letters_weights.pth",
        "csv": "latent_emnist_letters.csv",
        "bounds": "latent_emnist_letters_bounds.json",
        "meta": "dataset_emnist_letters.json",
        "label": "letter",
    },
    "fashion-mnist": {
        "weights": "vae_fashion_mnist_weights.pth",
        "csv": "latent_fashion_mnist.csv",
        "bounds": "latent_fashion_mnist_bounds.json",
        "meta": "dataset_fashion_mnist.json",
        "label": "class",
    },
    "kmnist": {
        "weights": "vae_kmnist_weights.pth",
        "csv": "latent_kmnist.csv",
        "bounds": "latent_kmnist_bounds.json",
        "meta": "dataset_kmnist.json",
        "label": "character",
    },
}

EMNIST_LETTERS = list(string.ascii_uppercase)  # index 0 -> 'A', ..., 25 -> 'Z'
FASHION_NAMES = [
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Boot",
]
KMNIST_NAMES = [f"kana-{i}" for i in range(10)]


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


def base_transform():
    """Standard 28x28 grayscale -> [0,1] tensor."""
    return transforms.Compose([transforms.ToTensor()])


def emnist_transform():
    """EMNIST comes transposed vs MNIST; transpose H/W back after ToTensor."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            # Tensor is (C, H, W); swap H and W to fix rotation.
            transforms.Lambda(lambda t: t.transpose(1, 2)),
        ]
    )


def load_dataset(name: str, max_train: int | None = None, batch_size: int = BATCH_SIZE):
    """Download/prepare a dataset; return (train_loader, test_dataset, classes)."""
    if name == "mnist":
        train_data = datasets.MNIST("./data", train=True, download=True, transform=base_transform())
        test_data = datasets.MNIST("./data", train=False, download=True, transform=base_transform())
        classes = [str(i) for i in range(10)]
    elif name == "emnist-letters":
        # split='letters': 124.8k train / 20.8k test, labels 1-26 (1=A).
        train_data = datasets.EMNIST(
            "./data", split="letters", train=True, download=True, transform=emnist_transform()
        )
        test_data = datasets.EMNIST(
            "./data", split="letters", train=False, download=True, transform=emnist_transform()
        )
        classes = EMNIST_LETTERS
    elif name == "fashion-mnist":
        train_data = datasets.FashionMNIST("./data", train=True, download=True, transform=base_transform())
        test_data = datasets.FashionMNIST("./data", train=False, download=True, transform=base_transform())
        classes = FASHION_NAMES
    elif name == "kmnist":
        train_data = datasets.KMNIST("./data", train=True, download=True, transform=base_transform())
        test_data = datasets.KMNIST("./data", train=False, download=True, transform=base_transform())
        classes = KMNIST_NAMES
    else:  # pragma: no cover - guarded by argparse choices
        raise ValueError(f"Unknown dataset: {name}. Choose from {DATASETS}")

    if max_train is not None and max_train < len(train_data):
        gen = torch.Generator().manual_seed(SEED)
        idx = torch.randperm(len(train_data), generator=gen)[:max_train].tolist()
        train_data = Subset(train_data, idx)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    print(f"Dataset: {name} | train images: {len(train_data)} | test images: {len(test_data)}")
    return train_loader, test_data, classes


def train_vae(
    model: VAE,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = EPOCHS,
    lr: float = LR,
) -> None:
    """Run the training loop, printing a per-epoch loss summary."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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
    csv_path: str,
    dataset: str,
    classes: list[str],
    n_samples: int = N_LATENT_SAMPLES,
) -> None:
    """Encode test images, save (x, y, label) rows to CSV.

    Encodes the full test set, then takes a stratified (class-balanced)
    sample of ``n_samples`` so every class shows up on the map. The old
    first-N approach broke on EMNIST Letters because its test set is
    grouped by letter (first 1000 rows were only A/B).
    """
    model.eval()

    loader = DataLoader(test_data, batch_size=512, shuffle=False)
    zs, labels = [], []
    for x, y in loader:
        x = x.to(device)
        mu, _ = model.encoder(x)  # use the latent mean as the representative point
        zs.append(mu.cpu())
        labels.append(y.to(torch.int64))

    z_all = torch.cat(zs, dim=0)
    y_all = torch.cat(labels, dim=0)
    if dataset == "emnist-letters":
        # EMNIST letters labels are 1-26; shift to 0-25 for plotting.
        y_all = y_all - 1

    # Stratified sample: same count per class, deterministic via SEED.
    n_classes = len(classes)
    per_class = max(1, n_samples // n_classes)
    gen = torch.Generator().manual_seed(SEED)
    keep = []
    for c in range(n_classes):
        idx = (y_all == c).nonzero(as_tuple=True)[0]
        if len(idx) == 0:
            continue
        perm = idx[torch.randperm(len(idx), generator=gen)[:per_class]]
        keep.append(perm)
    keep = torch.cat(keep)
    # Shuffle the kept rows so the CSV isn't grouped by class.
    keep = keep[torch.randperm(len(keep), generator=gen)]

    z = z_all[keep]
    y = y_all[keep]

    y_np = y.numpy().astype(int)
    if dataset == "emnist-letters":
        names = [classes[i] for i in y_np]
    elif dataset == "mnist":
        names = [str(i) for i in y_np]
    else:
        names = [classes[i] if 0 <= i < len(classes) else str(i) for i in y_np]

    df = pd.DataFrame(
        {
            "latent_x": z[:, 0].numpy(),
            "latent_y": z[:, 1].numpy(),
            "label": y_np,
            "label_name": names,
            # Legacy column so the old digit-only app still works.
            "digit": y_np,
        }
    )
    df.to_csv(csv_path, index=False)
    print(f"Saved latent coordinates for {len(df)} images -> {csv_path}")
    print(f"Latent ranges: x [{df.latent_x.min():.2f}, {df.latent_x.max():.2f}], "
          f"y [{df.latent_y.min():.2f}, {df.latent_y.max():.2f}]")


def parse_args():
    p = argparse.ArgumentParser(description="Train a 2D-latent Conv VAE on MNIST or an alphabet dataset.")
    p.add_argument("--dataset", choices=DATASETS, default="mnist",
                   help="mnist (digits) or emnist-letters (A-Z alphabet). Default: mnist")
    p.add_argument("--epochs", type=int, default=EPOCHS, help=f"Training epochs (default {EPOCHS})")
    p.add_argument("--max-train", type=int, default=None,
                   help="Cap training images for a quick run, e.g. 20000. Default: full dataset")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--n-samples", type=int, default=N_LATENT_SAMPLES,
                   help="Test images to project into latent_data CSV")
    p.add_argument("--weights", default=None, help="Override weights output path")
    p.add_argument("--csv", default=None, help="Override latent CSV output path")
    return p.parse_args()


def main(args=None) -> None:
    args = args or parse_args()
    set_seed(SEED)
    device = get_device()
    print(f"Using device: {device}")

    files = DATASET_FILES[args.dataset]
    weights_path = args.weights or files["weights"]
    csv_path = args.csv or files["csv"]
    bounds_path = files["bounds"]
    meta_path = files["meta"]

    train_loader, test_data, classes = load_dataset(args.dataset, max_train=args.max_train, batch_size=args.batch_size)

    model = VAE(z_dim=Z_DIM).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    train_vae(model, train_loader, device, epochs=args.epochs, lr=args.lr)

    # ---- Save trained weights -------------------------------------------
    torch.save(model.state_dict(), weights_path)
    print(f"Saved model weights -> {weights_path}")

    # ---- Extract and save the 2D latent coordinates ----------------------
    build_latent_csv(model, test_data, device, csv_path, args.dataset, classes,
                     n_samples=args.n_samples)

    # ---- Also dump a small sidecar describing the latent-space bounds so
    # ---- the dashboard can align its sliders with the scatter plot. -----
    df = pd.read_csv(csv_path)
    bounds = {
        "x_min": float(df.latent_x.min()),
        "x_max": float(df.latent_x.max()),
        "y_min": float(df.latent_y.min()),
        "y_max": float(df.latent_y.max()),
    }
    with open(bounds_path, "w") as fh:
        json.dump(bounds, fh, indent=2)
    print(f"Saved latent bounds -> {bounds_path}")

    meta = {"dataset": args.dataset, "classes": classes, "label_column": "label",
            "weights": weights_path, "csv": csv_path, "bounds": bounds_path,
            "epochs": args.epochs}
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"Saved dataset meta -> {meta_path}")


if __name__ == "__main__":
    main()