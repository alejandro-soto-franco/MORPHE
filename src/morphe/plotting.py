"""Shared plotting helpers for training curves.

Ported from the various ``plot_loss``/``_plot_loss_curve`` cells; saves to a
file instead of calling ``plt.show()``, since these run outside a notebook.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_loss(train_losses: list[float], val_losses: list[float], out_path: str | Path) -> None:
    """Plot train/validation loss curves and save to ``out_path``."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(train_losses, label="train")
    ax.plot(val_losses, label="val")
    ax.legend()
    ax.grid(True)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss curve")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
