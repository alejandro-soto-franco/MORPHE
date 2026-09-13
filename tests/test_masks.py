"""Regression tests for the mask-axis-convention bug (see README).

``mask[:, x1:x2, y1:y2]`` on a ``[C, H, W]`` tensor puts the bbox's x-extent
(scaled by width) on the row (height) axis, and the y-extent (scaled by
height) on the column (width) axis. These tests pin down both the
reproduced upstream behaviour (``"upstream"``, needed for the published
checkpoints) and the corrected behaviour (``"xy"``).

Bbox values are exact binary fractions (0.25, 0.75) so their scaled extents
are unambiguous in both float32 and float64 arithmetic; arbitrary decimals
like 0.4 or 0.9 round slightly differently depending on whether ``.item()``
is called before or after the multiply, which made an earlier version of
this file flaky.
"""

import pytest
import torch

from morphe.masks import bbox_to_pixel_mask, create_latent_mask

HEIGHT, WIDTH = 10, 20
# x-extent (scaled by WIDTH=20): 0.25 -> 5. y-extent (scaled by HEIGHT=10): 0.75 -> 7.
BBOX = torch.tensor([0.0, 0.0, 0.25, 0.75])
X_EXTENT = 5
Y_EXTENT = 7


def test_upstream_convention_is_transposed():
    mask = bbox_to_pixel_mask(BBOX, HEIGHT, WIDTH, convention="upstream")

    assert mask.shape == (1, HEIGHT, WIDTH)
    filled_rows = (mask[0].sum(dim=1) > 0).nonzero().flatten()
    filled_cols = (mask[0].sum(dim=0) > 0).nonzero().flatten()

    # Reproduced defect: the x-extent (5) lands on the row axis, the
    # y-extent (7) lands on the column axis.
    assert filled_rows.numel() == X_EXTENT
    assert filled_cols.numel() == Y_EXTENT


def test_xy_convention_is_correct():
    mask = bbox_to_pixel_mask(BBOX, HEIGHT, WIDTH, convention="xy")

    filled_rows = (mask[0].sum(dim=1) > 0).nonzero().flatten()
    filled_cols = (mask[0].sum(dim=0) > 0).nonzero().flatten()

    # Corrected: the y-extent (7) lands on the row axis, the x-extent (5)
    # lands on the column axis.
    assert filled_rows.numel() == Y_EXTENT
    assert filled_cols.numel() == X_EXTENT


def test_create_latent_mask_transposes_the_same_way_as_pixel_mask():
    # Production latents are always square (SD1.5 downsamples 512x512 to
    # 64x64), so use a square shape here: create_latent_mask's "upstream"
    # meshgrid path returns a (W, H)-shaped mask per bbox, which only equals
    # the (H, W) shape used elsewhere in the codebase when H == W.
    size = 16
    bbox = BBOX.unsqueeze(0)
    shape = (1, 4, size, size)
    x_extent = int(0.25 * size)  # 4
    y_extent = int(0.75 * size)  # 12

    for convention in ("upstream", "xy"):
        latent_mask = create_latent_mask(bbox, shape, device="cpu", convention=convention)[0, 0]
        pixel_mask = bbox_to_pixel_mask(bbox[0], size, size, convention=convention)[0]

        assert latent_mask.shape == (size, size)

        # pixel_mask slices (exclusive of the upper bound); latent_mask
        # compares against a threshold (inclusive), matching the two
        # different upstream implementations exactly -- so latent_mask's
        # extent along the filled axis is one larger.
        for mask, extent_bump in ((pixel_mask, 0), (latent_mask, 1)):
            filled_rows = (mask.sum(dim=1) > 0).nonzero().flatten()
            filled_cols = (mask.sum(dim=0) > 0).nonzero().flatten()
            if convention == "upstream":
                assert filled_rows.numel() == x_extent + extent_bump
                assert filled_cols.numel() == y_extent + extent_bump
            else:
                assert filled_rows.numel() == y_extent + extent_bump
                assert filled_cols.numel() == x_extent + extent_bump


def test_create_latent_mask_upstream_is_shape_transposed_on_nonsquare_latents():
    # The upstream defect is not only value-transposed but literally
    # shape-transposed for a non-square latent grid: torch.meshgrid(arange(W),
    # arange(H)) at its default 'ij' indexing returns (W, H)-shaped tensors,
    # not the (H, W) latent_shape declares. This never surfaces in the
    # original codebase because every real latent there is square.
    bbox = BBOX.unsqueeze(0)
    shape = (1, 4, HEIGHT, WIDTH)

    latent_mask = create_latent_mask(bbox, shape, device="cpu", convention="upstream")

    assert latent_mask.shape == (1, 1, WIDTH, HEIGHT)


def test_unknown_convention_rejected():
    with pytest.raises(ValueError):
        bbox_to_pixel_mask(torch.tensor([0.0, 0.0, 0.5, 0.5]), 10, 10, convention="bogus")  # type: ignore[arg-type]
