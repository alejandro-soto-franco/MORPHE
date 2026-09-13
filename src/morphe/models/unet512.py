"""Stage-2 512x512 conditional UNet for the pixel-diffusion decoder.

Ported unchanged from ``Pixel_Diffusion_Decoder/models/unet512.py``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Sinusoidal timestep embedding, as used by ``diffusers``.

    ``timesteps``: ``[B]``. Returns ``[B, dim]``.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)

    if dim % 2:
        emb = torch.cat([emb, emb[:, :1]], dim=1)
    return emb


class ResidualBlock2d(nn.Module):
    """Conv-GN-SiLU-Conv block with an additive timestep projection."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int | None = None):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.act = nn.SiLU()

        self.use_time = time_dim is not None
        if self.use_time:
            # pyrefly: ignore [bad-argument-type]
            self.time_proj = nn.Linear(time_dim, out_ch)

        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor | None = None) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))

        if self.use_time:
            h = h + self.time_proj(self.act(t_emb))[:, :, None, None]

        h = self.conv2(self.act(self.norm2(h)))
        return h + self.shortcut(x)


class ViTAttention(nn.Module):
    """Standard multi-head self-attention, written out to avoid a ``timm`` dependency."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``[B, N, C]``."""
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TimmAttn2D(nn.Module):
    """2D wrapper over :class:`ViTAttention`: ``[B, C, H, W]`` -> flatten -> attend -> reshape."""

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(8, dim)
        self.attn = ViTAttention(dim, num_heads=num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        h = h.flatten(2).transpose(1, 2)
        h = self.attn(h)
        h = h.transpose(1, 2).reshape(B, C, H, W)
        return h


class Down(nn.Module):
    """Two residual blocks (with optional concatenated conditioning) then avg-pool."""

    def __init__(self, in_ch: int, out_ch: int, tdim: int, cond_ch: int = 0):
        super().__init__()
        self.block1 = ResidualBlock2d(in_ch + cond_ch, out_ch, time_dim=tdim)
        self.block2 = ResidualBlock2d(out_ch, out_ch, time_dim=tdim)
        self.down = nn.AvgPool2d(2)

    def forward(
        self, x: torch.Tensor, t_emb: torch.Tensor, cond: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if cond is not None:
            x = torch.cat([x, cond], dim=1)

        h = self.block1(x, t_emb)
        h = self.block2(h, t_emb)
        return h, self.down(h)


class Up(nn.Module):
    """Bilinear 2x upsample, concat skip (+ optional conditioning), two residual blocks."""

    def __init__(self, in_ch: int, out_ch: int, tdim: int, cond_ch: int = 0):
        super().__init__()
        self.block1 = ResidualBlock2d(in_ch + cond_ch, out_ch, time_dim=tdim)
        self.block2 = ResidualBlock2d(out_ch, out_ch, time_dim=tdim)

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        t_emb: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        if cond is not None:
            x = torch.cat([x, cond], dim=1)

        x = self.block1(x, t_emb)
        x = self.block2(x, t_emb)
        return x


class UNet512(nn.Module):
    """Stage-2 512x512 UNet: noisy RGB image + timestep + multi-scale conditioning -> RGB."""

    def __init__(self, base_ch: int = 128, cond_ch: int = 64, time_dim: int = 256):
        super().__init__()

        ch1, ch2, ch3 = base_ch, base_ch * 2, base_ch * 4

        self.time_mlp = nn.Sequential(
            nn.Linear(320, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.in_conv = nn.Conv2d(3 + cond_ch, ch1, 3, padding=1)

        self.down1 = Down(ch1, ch1, tdim=time_dim, cond_ch=0)
        self.down2 = Down(ch1, ch2, tdim=time_dim, cond_ch=cond_ch)
        self.down3 = Down(ch2, ch3, tdim=time_dim, cond_ch=cond_ch)

        self.attn64 = TimmAttn2D(dim=ch3, num_heads=4)
        self.mid1 = ResidualBlock2d(ch3 + cond_ch, ch3, time_dim=time_dim)
        self.mid2 = ResidualBlock2d(ch3, ch3, time_dim=time_dim)

        self.up3 = Up(ch3 + ch3, ch2, tdim=time_dim, cond_ch=cond_ch)
        self.up2 = Up(ch2 + ch2, ch1, tdim=time_dim, cond_ch=cond_ch)
        self.up1 = Up(ch1 + ch1, ch1, tdim=time_dim, cond_ch=0)

        self.out_norm = nn.GroupNorm(8, ch1)
        self.out = nn.Conv2d(ch1, 3, 3, padding=1)

    def forward(
        self, x: torch.Tensor, timesteps: torch.Tensor, cond_feats: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """``x``: ``[B,3,512,512]``. ``cond_feats``: ``{s64, s128, s256, s512}``."""
        t_emb = self.time_mlp(timestep_embedding(timesteps, 320))

        x = torch.cat([x, cond_feats["s512"]], dim=1)
        x = self.in_conv(x)

        skip1, x = self.down1(x, t_emb, cond=None)
        skip2, x = self.down2(x, t_emb, cond_feats["s256"])
        skip3, x = self.down3(x, t_emb, cond_feats["s128"])

        x = self.attn64(x)
        x = torch.cat([x, cond_feats["s64"]], dim=1)
        x = self.mid1(x, t_emb)
        x = self.mid2(x, t_emb)

        x = self.up3(x, skip3, t_emb, cond_feats["s128"])
        x = self.up2(x, skip2, t_emb, cond_feats["s256"])
        x = self.up1(x, skip1, t_emb, cond=None)

        x = self.out(self.out_norm(x).clamp(-6, 6))
        return torch.tanh(x)
