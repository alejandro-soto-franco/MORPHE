"""Bbox-coordinate encoder for the outpainting / 2D-imputation UNet.

The upstream ``Train/models/coord_encoder.py`` defines this as a class
wrapping the layers in ``self.net`` (``CoordEncoder(nn.Module)`` with
``self.net = nn.Sequential(...)``), but the actually-published checkpoint
(``Hickey-Lab/MORPHE_CODEX_Outpainting``'s ``model_2.safetensors``) has flat
keys (``0.weight``, ``0.bias``), matching the *inference* notebooks' inline
``nn.Sequential(nn.Linear(4, 32), nn.GELU())`` instead. The training run that
produced the published weights evidently used that form, not the
``Train/models/coord_encoder.py`` class; :class:`CoordEncoder` subclasses
``nn.Sequential`` directly here to match the real checkpoint layout.
"""

from __future__ import annotations

import torch.nn as nn


class CoordEncoder(nn.Sequential):
    """Maps a ``[B, 4]`` normalised bbox to a ``[B, dim]`` embedding."""

    def __init__(self, dim: int = 32):
        super().__init__(nn.Linear(4, dim), nn.GELU())
