"""
model.py
========
Lightweight Convolutional Variational Autoencoder (VAE) for MNIST.

The model compresses a 28x28 grayscale image into a 2-dimensional latent
vector ``z`` and reconstructs it back. Because ``z`` has only two dimensions
we can visualise the entire latent space as a 2D scatter plot, which is the
core idea behind the Interactive VAE Latent Space Explorer.

Components
----------
Encoder : Conv layers -> (mu, log_var) of the 2D latent Gaussian
Decoder : Fully-connected -> ConvTranspose layers -> 28x28 image
vae_loss: Reconstruction loss (BCE) + KL divergence
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# The latent space is intentionally 2D so it can be plotted and explored.
Z_DIM = 2


class Encoder(nn.Module):
    """Maps a 1x28x28 image to the parameters (mu, log_var) of a 2D Gaussian."""

    def __init__(self, z_dim: int = Z_DIM):
        super().__init__()
        self.z_dim = z_dim

        # 1x28x28 -> 32x14x14
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        # 32x14x14 -> 64x7x7
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)

        # 64x7x7 -> flattened feature vector
        self.fc = nn.Linear(64 * 7 * 7, 128)
        # Two heads: mean and log-variance of the latent distribution.
        self.fc_mu = nn.Linear(128, z_dim)
        self.fc_logvar = nn.Linear(128, z_dim)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc(x))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    """Maps a 2D latent vector back to a 1x28x28 reconstructed image."""

    def __init__(self, z_dim: int = Z_DIM):
        super().__init__()
        self.z_dim = z_dim

        # Latent vector -> spatial feature map 64x7x7
        self.fc = nn.Linear(z_dim, 128)
        self.fc2 = nn.Linear(128, 64 * 7 * 7)

        # 64x7x7 -> 32x14x14 -> 1x28x28
        self.deconv1 = nn.ConvTranspose2d(
            64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.deconv2 = nn.ConvTranspose2d(
            32, 1, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.bn1 = nn.BatchNorm2d(32)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc(z))
        x = F.relu(self.fc2(x))
        x = x.view(-1, 64, 7, 7)
        x = F.relu(self.bn1(self.deconv1(x)))
        # Sigmoid keeps pixel values in [0, 1] to match BCE reconstruction loss.
        x = torch.sigmoid(self.deconv2(x))
        return x


class VAE(nn.Module):
    """Full VAE wrapping the encoder and decoder with the reparameterisation trick."""

    def __init__(self, z_dim: int = Z_DIM):
        super().__init__()
        self.z_dim = z_dim
        self.encoder = Encoder(z_dim)
        self.decoder = Decoder(z_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample z ~ N(mu, sigma^2) using the reparameterisation trick.

        z = mu + sigma * eps, with eps ~ N(0, I). This keeps the sampling
        operation differentiable so gradients can flow to the encoder.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Deterministic decode of a latent vector (used by the Streamlit app)."""
        return self.decoder(z)


def vae_loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float = 1.0,
):
    """Standard VAE loss = Reconstruction (BCE) + KL divergence.

    Args:
        recon: reconstructed images, shape (N, 1, 28, 28), values in [0, 1].
        x:     original images, shape (N, 1, 28, 28), values in [0, 1].
        mu:    latent means, shape (N, z_dim).
        logvar: latent log-variances, shape (N, z_dim).
        kl_weight: scaling factor for the KL term (beta-VAE style).

    Returns:
        (total_loss, recon_loss, kl_loss) as scalar tensors.
    """
    # Sum BCE over all pixels, average over the batch.
    recon_loss = F.binary_cross_entropy(recon, x, reduction="sum") / x.size(0)
    # KL divergence of N(mu, sigma^2) from the standard normal prior N(0, I).
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    total = recon_loss + kl_weight * kl_loss
    return total, recon_loss, kl_loss


if __name__ == "__main__":
    # Quick self-test: run a random batch through the VAE and the loss.
    net = VAE(z_dim=2)
    dummy = torch.rand(4, 1, 28, 28)
    recon, mu, logvar = net(dummy)
    loss, r, k = vae_loss(recon, dummy, mu, logvar)
    print("Latent dim     :", net.z_dim)
    print("Recon shape    :", tuple(recon.shape))
    print("mu/logvar shape:", tuple(mu.shape), tuple(logvar.shape))
    print(f"total={loss.item():.2f} recon={r.item():.2f} kl={k.item():.2f}")
