"""Conditioning encoder for the outpainting / 2D-imputation UNet.

Ported unchanged (architecture and defaults) from
``Latent_Diffusion_Generator/Outpainting&2d-Imputation/Train/models/cond_encoder.py``.
Encodes the masked-region latent into a token sequence the UNet attends to
as ``encoder_hidden_states``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class Transpose(nn.Module):
    """Swap two dimensions; used to turn a ``[B, C, N]`` map into token order."""

    def __init__(self, dim0: int, dim1: int):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(self.dim0, self.dim1)


class PositionEncoding2D(nn.Module):
    """Fixed sinusoidal positional encoding added to a token sequence."""

    def __init__(self, num_patches: int, dim: int):
        super().__init__()
        self.register_buffer("pos_embed", self.build(num_patches, dim), persistent=False)

    def build(self, num_patches: int, dim: int) -> torch.Tensor:
        pe = torch.zeros(num_patches, dim)
        pos = torch.arange(num_patches).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        return pe.unsqueeze(0)  # [1, num_patches, dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos_embed[:, : x.size(1)]


class ResidualBlock(nn.Module):
    """Conv-GroupNorm-SiLU residual block with a 1x1 skip projection."""

    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.GroupNorm(8, out_c),
            nn.SiLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.GroupNorm(8, out_c),
        )
        self.skip = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.skip(x)


class CondEncoder(nn.Module):
    """Encodes a ``[B, in_channels, H, W]`` latent into ``num_tokens`` embed vectors."""

    def __init__(self, in_channels: int = 4, embed_dim: int = 736, num_tokens: int = 64):
        super().__init__()

        self.encoder = nn.Sequential(
            ResidualBlock(in_channels, 64),
            nn.AvgPool2d(2),
            ResidualBlock(64, 128),
            nn.AvgPool2d(2),
            ResidualBlock(128, 256),
            nn.AvgPool2d(2),
            nn.Conv2d(256, embed_dim, 1),
        )

        self.proj = nn.Sequential(nn.Flatten(2), Transpose(-1, -2))

        self.pos = PositionEncoding2D(num_tokens, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)  # [B, embed_dim, 8, 8]
        tokens = self.proj(feat)  # [B, 64, embed_dim]
        tokens = self.pos(tokens)
        tokens = self.norm(tokens)
        return tokens
