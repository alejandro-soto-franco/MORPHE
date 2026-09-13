"""Compresses per-cell classification probabilities into an RGB-like embedding.

Ported from ``Embeddings/02_Autoencoder.ipynb``. The trained encoder's
3-dimensional bottleneck is min-max scaled to ``[0, 255]`` and written as an
RGB image per tissue region; that image is what the latent-diffusion stage
is trained and conditioned on.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split


class Autoencoder(nn.Module):
    """A five-layer bottleneck autoencoder, ``in_dim`` -> ``bottleneck_dim`` -> ``in_dim``."""

    def __init__(self, in_dim: int = 25, bottleneck_dim: int = 3, hidden_dim: int = 512):
        super().__init__()

        def block(a: int, b: int) -> list[nn.Module]:
            return [nn.Linear(a, b), nn.ReLU(), nn.LayerNorm(b), nn.Dropout(0.1)]

        h4, h2 = hidden_dim // 4, hidden_dim // 2
        self.encoder = nn.Sequential(
            *block(in_dim, h4),
            *block(h4, h2),
            *block(h2, hidden_dim),
            *block(hidden_dim, h2),
            *block(h2, h4),
            nn.Linear(h4, bottleneck_dim),
        )
        self.decoder = nn.Sequential(
            *block(bottleneck_dim, h4),
            *block(h4, h2),
            *block(h2, hidden_dim),
            *block(hidden_dim, h2),
            *block(h2, h4),
            nn.Linear(h4, in_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return z, self.decoder(z)


def bio_contrastive_loss(
    z: torch.Tensor, orig_probs: torch.Tensor, margin: float = 50.0, alpha: float = 0.1
) -> torch.Tensor:
    """Pulls same-type embeddings together and pushes different types apart.

    The inter-class penalty is scaled down between biologically similar
    types (measured by cosine similarity of their probability vectors), so
    similar cell types need not be pushed as far apart in the bottleneck.
    """
    dists = torch.cdist(z, z, p=2)

    labels = orig_probs.argmax(dim=1)
    same = labels.unsqueeze(1) == labels.unsqueeze(0)
    diff = ~same

    prob_sim = F.cosine_similarity(orig_probs.unsqueeze(1), orig_probs.unsqueeze(0), dim=-1)
    prob_sim = torch.clamp(prob_sim, 0, 1)

    intra_loss = dists[same].sum() / (same.sum() + 1e-8)

    inter_weight = 1 - alpha * prob_sim[diff]
    inter_loss = (inter_weight * torch.clamp(margin - dists[diff], min=0)).sum() / (diff.sum() + 1e-8)

    return 0.1 * intra_loss + inter_loss


def loss_function(
    orig_probs: torch.Tensor,
    pred_logits: torch.Tensor,
    z: torch.Tensor,
    margin: float = 50.0,
    beta: float = 0.1,
) -> tuple[torch.Tensor, float, float]:
    pred_log_probs = F.log_softmax(pred_logits, dim=1)
    recon_loss = F.kl_div(pred_log_probs, orig_probs, reduction="batchmean")
    cluster_loss = bio_contrastive_loss(z, orig_probs, margin)
    total_loss = recon_loss + beta * cluster_loss
    return total_loss, recon_loss.item(), cluster_loss.item()


def train_autoencoder(
    prob_csv: str | Path,
    checkpoint_path: str | Path,
    epochs: int = 20,
    batch_size: int = 4096,
    lr: float = 1e-5,
    val_ratio: float = 0.1,
    alpha: float = 0.1,
    seed: int = 0,
    device: torch.device | str = "cpu",
) -> tuple[Autoencoder, pd.DataFrame]:
    """Train the autoencoder on the GCN classifier's per-class probability columns."""
    result_df = pd.read_csv(prob_csv)
    prob_cols = [c for c in result_df.columns if c.startswith("prob_")]
    emb_matrix = result_df[prob_cols].values
    emb_matrix_tensor = torch.tensor(emb_matrix, dtype=torch.float32)

    dataset = TensorDataset(emb_matrix_tensor)
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = Autoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for (x,) in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            z, out = model(x)
            loss, _, _ = loss_function(x, out, z, beta=alpha)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss / len(train_loader)

        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for (x,) in val_loader:
                x = x.to(device)
                z, out = model(x)
                loss, _, _ = loss_function(x, out, z, beta=alpha)
                val_total += loss.item()
        val_loss = val_total / max(1, len(val_loader))

        print(f"epoch {epoch + 1}/{epochs} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f}")

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)

    return model, result_df


def embed_to_rgb(
    model: Autoencoder, result_df: pd.DataFrame, device: torch.device | str = "cpu"
) -> pd.DataFrame:
    """Encode every row's probability vector and append min-max scaled R/G/B columns."""
    prob_cols = [c for c in result_df.columns if c.startswith("prob_")]
    emb_matrix = result_df[prob_cols].values

    model.eval()
    with torch.no_grad():
        x_in = torch.tensor(emb_matrix, dtype=torch.float32).to(device)
        z_3d = model.encoder(x_in).cpu().numpy()

    min_vals = z_3d.min(axis=0)
    max_vals = z_3d.max(axis=0)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1e-9

    scaled_3d = (z_3d - min_vals) / range_vals
    rgb_3d = (scaled_3d * 255).astype(np.uint8)

    out = result_df.copy()
    out["R"] = rgb_3d[:, 0]
    out["G"] = rgb_3d[:, 1]
    out["B"] = rgb_3d[:, 2]
    return out


def export_region_images(rgb_df: pd.DataFrame, save_dir: str | Path, image_size: int = 1024) -> list[Path]:
    """Rasterise each region's (x, y, R, G, B) rows into an RGB PNG."""
    import cv2

    save_dir = Path(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    paths = []
    for cnt, region in enumerate(rgb_df["region"].unique()):
        subset = rgb_df[rgb_df["region"] == region]
        img = np.ones((image_size, image_size, 3), dtype=np.uint8) * 255

        cols = zip(subset["x"], subset["y"], subset["R"], subset["G"], subset["B"], strict=True)
        for x, y, r, g, b in cols:
            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < image_size and 0 <= yi < image_size:
                img[yi, xi, 0] = b
                img[yi, xi, 1] = g
                img[yi, xi, 2] = r

        out_path = save_dir / f"region_{cnt}.png"
        cv2.imwrite(str(out_path), img)
        paths.append(out_path)

    return paths
