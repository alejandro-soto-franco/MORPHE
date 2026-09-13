"""Bounding-box masks in pixel and latent space.

Upstream defect (see ``Differences from upstream`` in the README): the
original ``mask[:, x1:x2, y1:y2]`` indexing on a ``[C, H, W]`` tensor, and the
matching default-``'ij'`` :func:`torch.meshgrid` call in the latent-space
mask builder, both put the bbox's x-extent on the row (height) axis and the
y-extent on the column (width) axis. That is transposed relative to the
bbox convention every dataset and the trainer otherwise use, where
``bbox = (x1, y1, x2, y2)`` normalised to width/height respectively.

The published HuggingFace checkpoints (``Hickey-Lab/MORPHE_CODEX_Outpainting``
and the 2D-imputation inference notebooks that share it) were trained with
the transposed convention, so silently correcting it would make the shipped
weights condition on the wrong region at inference time. ``mask_axis_convention``
selects between the two: ``"upstream"`` (default) reproduces the original,
transposed behaviour so the published weights stay valid; ``"xy"`` applies
the bbox in the conventional x-is-width, y-is-height sense, for anyone
training a fresh checkpoint.
"""

from __future__ import annotations

from typing import Literal

import torch

MaskAxisConvention = Literal["upstream", "xy"]


def bbox_to_pixel_mask(
    bbox: torch.Tensor,
    height: int,
    width: int,
    convention: MaskAxisConvention = "upstream",
) -> torch.Tensor:
    """Build a ``[1, height, width]`` binary mask from a normalised bbox.

    ``bbox`` is ``(x1, y1, x2, y2)`` with each component normalised to
    ``[0, 1]`` against (width, height, width, height) respectively, matching
    :meth:`morphe.datasets.stage1_dataset.Stage1Dataset._gen_random_bbox`.
    """
    if bbox.numel() != 4:
        raise ValueError(f"expected a 4-element bbox, got shape {tuple(bbox.shape)}")

    x1 = int(bbox[0].item() * width)
    y1 = int(bbox[1].item() * height)
    x2 = int(bbox[2].item() * width)
    y2 = int(bbox[3].item() * height)

    mask = torch.zeros(1, height, width, dtype=torch.float32)
    if convention == "upstream":
        # Reproduces the original defect: bbox-x indexes the row axis.
        mask[:, x1:x2, y1:y2] = 1.0
    elif convention == "xy":
        mask[:, y1:y2, x1:x2] = 1.0
    else:
        raise ValueError(f"unknown mask_axis_convention {convention!r}")
    return mask


def create_latent_mask(
    bbox: torch.Tensor,
    latent_shape: tuple[int, int, int, int],
    device: torch.device | str,
    convention: MaskAxisConvention = "upstream",
) -> torch.Tensor:
    """Build a batch of latent-space masks, one bbox per row of ``bbox``.

    ``latent_shape`` is ``(B, C, H, W)`` as returned by ``vae.encode(...)``.
    Returns a ``[B, 1, H, W]`` mask.
    """
    _, _, height, width = latent_shape
    masks = []

    for coords in bbox:
        x1, y1, x2, y2 = coords * torch.tensor(
            [width, height, width, height], device=device, dtype=coords.dtype
        )
        if convention == "upstream":
            # Original defect: meshgrid(W, H) with default 'ij' indexing puts
            # the width-range on dim 0, so the resulting mask is [W, H] where
            # a latent tensor's spatial dims are [H, W].
            xx, yy = torch.meshgrid(
                torch.arange(width, device=device),
                torch.arange(height, device=device),
            )
            mask = ((xx >= x1) & (xx <= x2) & (yy >= y1) & (yy <= y2)).float()
        elif convention == "xy":
            yy, xx = torch.meshgrid(
                torch.arange(height, device=device),
                torch.arange(width, device=device),
                indexing="ij",
            )
            mask = ((xx >= x1) & (xx <= x2) & (yy >= y1) & (yy <= y2)).float()
        else:
            raise ValueError(f"unknown mask_axis_convention {convention!r}")
        masks.append(mask)

    return torch.stack(masks).unsqueeze(1)
