"""Mask-conditioned coordinate encoder for arbitrary-shape inpainting.

Ported from ``Latent_Diffusion_Generator/Arbitrary_Inpainting/Train_Arbitrary_Inpainting.ipynb``.
Distinct from :class:`morphe.models.coord_encoder.CoordEncoder` (which
encodes a 4-value bbox): this one encodes a full binary mask, since arbitrary
inpainting conditions on irregular polygon masks rather than rectangles.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskCoordEncoder(nn.Module):
    """Downsamples a full-resolution mask to 64x64 and projects it to token embeddings."""

    def __init__(self, embed_dim: int = 32, num_tokens: int = 64):
        super().__init__()
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim

        self.mlp = nn.Sequential(
            nn.Linear(64 * 64, 2048),
            nn.GELU(),
            nn.Linear(2048, num_tokens * embed_dim),
        )

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        mask_ds = F.interpolate(mask, size=(64, 64), mode="nearest")
        B = mask_ds.shape[0]
        x = mask_ds.view(B, -1)
        x = self.mlp(x)
        return x.view(B, self.num_tokens, self.embed_dim)
