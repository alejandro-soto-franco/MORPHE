"""Decodes an RGB embedding image back into a per-pixel cell-type map.

Ported from ``Embeddings/03_Interpret_Cellmap.ipynb``.

The repository has two, inconsistent copies of ``infer_cell_map``: this one
(``Embeddings/03_Interpret_Cellmap.ipynb``), which inverts the RGB image back
to the autoencoder's latent scale before decoding and returns 1-indexed cell
types, and a second one duplicated into ``Evaluation/Evaluation.ipynb``, which
fed the raw ``[0, 1]`` RGB directly into the decoder (skipping the inverse
scaling) and returned 0-indexed types. Only the first is dimensionally
consistent with how :func:`morphe.embeddings.autoencoder.embed_to_rgb` builds
the image, so it is the version this module (and
:mod:`morphe.evaluation.metrics`) ports.

``z_min``/``z_range`` are the per-channel min/max the training run's encoder
output spanned before min-max scaling to ``[0, 255]``; they must be recorded
for each trained autoencoder checkpoint (printed by
:func:`morphe.embeddings.autoencoder.embed_to_rgb`) rather than reused across
checkpoints.
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from morphe.embeddings.autoencoder import Autoencoder

# Recorded from the original training run's encoder output range (see
# Embeddings/02_Autoencoder.ipynb cell 6). A fresh checkpoint must record and
# pass its own values.
DEFAULT_Z_MIN = (-69.761505, -75.65188, -77.16103)
DEFAULT_Z_MAX = (88.969406, 65.244896, 67.13518)


def load_and_recover_z3d_png(path: str | Path, white_threshold: int = 245) -> torch.Tensor:
    """Load an RGB embedding PNG as a ``[3, H, W]`` float tensor in ``[0, 1]``.

    Near-white pixels (all channels above ``white_threshold``) are snapped to
    pure white, matching the background convention ``embed_to_rgb`` produces.
    """
    img = Image.open(path).convert("RGB")
    arr = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
    rgb = arr.view(img.size[1], img.size[0], 3).permute(2, 0, 1).clone()

    mask = (rgb > white_threshold).all(dim=0)
    rgb[:, mask] = 255

    return rgb.float() / 255.0


def infer_cell_map(
    latent_image: torch.Tensor,
    model: Autoencoder,
    z_min: tuple[float, float, float] = DEFAULT_Z_MIN,
    z_max: tuple[float, float, float] = DEFAULT_Z_MAX,
) -> torch.Tensor:
    """Decode a ``[3, H, W]`` RGB embedding into a ``[1, H, W]`` cell-type map.

    Pure-white pixels (background) map to type 0; every other pixel is
    inverse-scaled back to the autoencoder's latent range and decoded, with
    predicted classes 1-indexed (``argmax + 1``) so 0 is reserved for
    background.
    """
    device = next(model.parameters()).device
    min_vals = torch.tensor(z_min, device=device)
    range_vals = torch.tensor(z_max, device=device) - min_vals

    H, W = latent_image.shape[1], latent_image.shape[2]
    latent_image = latent_image.to(device)

    flat_img = latent_image.permute(1, 2, 0).reshape(-1, 3)
    white_mask = (flat_img == 1.0).all(dim=1)
    infer_input_rgb = flat_img[~white_mask]

    pred = torch.zeros(flat_img.shape[0], dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        if infer_input_rgb.shape[0] > 0:
            z_recovered = infer_input_rgb * range_vals + min_vals
            logits = model.decoder(z_recovered)
            pred[~white_mask] = torch.argmax(logits, dim=1) + 1

    return pred.reshape(1, H, W)
