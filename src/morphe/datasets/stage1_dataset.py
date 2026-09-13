"""Outpainting / 2D-imputation training dataset.

Ported from ``Latent_Diffusion_Generator/Outpainting&2d-Imputation/Train/data/stage1_dataset.py``.

Two upstream defects, both listed under ``Differences from upstream``:

1. The file listing used ``os.listdir`` unfiltered, so
   ``trainers/train_data/1`` and ``trainers/train_data/111`` (two stray text
   files, since deleted) were opened as images. :class:`Stage1Dataset` now
   filters by image extension.
2. The mask written into ``[C, H, W]`` used ``mask[:, x1:x2, y1:y2]``, which
   puts the bbox's x-extent on the row axis. Fixed via
   :func:`morphe.masks.bbox_to_pixel_mask`, gated by ``mask_axis_convention``
   so inference against the published checkpoints (trained under the
   original, transposed convention) is unaffected.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

from morphe.masks import MaskAxisConvention, bbox_to_pixel_mask

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def list_image_files(root_dir: str | Path) -> list[Path]:
    """List image files directly under ``root_dir``, sorted for determinism."""
    root_dir = Path(root_dir)
    return sorted(p for p in root_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def gen_random_bbox(rng: random.Random | None = None) -> torch.Tensor:
    """Sample a normalised ``(x1, y1, x2, y2)`` bbox as in the upstream code."""
    # pyrefly: ignore [bad-assignment]
    rng = rng or random
    # pyrefly: ignore [missing-attribute]
    if rng.random() < 0.5:
        # pyrefly: ignore [missing-attribute]
        aspect_ratio = rng.uniform(1, 33)
    else:
        # pyrefly: ignore [missing-attribute]
        aspect_ratio = rng.uniform(0.03, 1)

    # pyrefly: ignore [missing-attribute]
    area = rng.uniform(0.1, 0.33) ** 2
    w = min(np.sqrt(area * aspect_ratio), 0.99)
    h = min(np.sqrt(area / aspect_ratio), 0.99)

    # pyrefly: ignore [missing-attribute]
    x1 = rng.uniform(0.05, 0.99 - w)
    # pyrefly: ignore [missing-attribute]
    y1 = rng.uniform(0.05, 0.99 - h)

    return torch.tensor([x1, y1, x1 + w, y1 + h], dtype=torch.float32)


class Stage1Dataset(Dataset):
    """Masked-image / target-image / bbox triples for stage-1 training.

    Parameters
    ----------
    root_dir:
        Directory of region images (one file per region).
    img_size:
        Unused directly (kept for parity with upstream's signature); images
        are used at their native resolution after the rotation transform.
    masks_per_image:
        Number of random masks sampled per image per epoch.
    mask_axis_convention:
        ``"upstream"`` (default) reproduces the transposed mask used to
        train the published checkpoints; ``"xy"`` applies the bbox in the
        conventional sense.
    """

    def __init__(
        self,
        root_dir: str | Path,
        img_size: int = 512,
        masks_per_image: int = 100,
        mask_axis_convention: MaskAxisConvention = "upstream",
        regions: tuple[str, ...] | None = None,
    ):
        self.img_files = list_image_files(root_dir)
        if regions is not None:
            wanted = set(regions)
            self.img_files = [p for p in self.img_files if p.stem in wanted]
        if not self.img_files:
            raise ValueError(f"no image files found under {root_dir} (regions={regions})")
        self.img_size = img_size
        self.masks_per_image = masks_per_image
        self.mask_axis_convention: MaskAxisConvention = mask_axis_convention

        self.transform = transforms.Compose(
            [
                transforms.RandomChoice(
                    [
                        transforms.Lambda(lambda x: x),
                        transforms.Lambda(lambda x: TF.rotate(x, 90)),
                        transforms.Lambda(lambda x: TF.rotate(x, 180)),
                        transforms.Lambda(lambda x: TF.rotate(x, 270)),
                    ]
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )

    def __len__(self) -> int:
        return len(self.img_files) * self.masks_per_image

    # pyrefly: ignore [bad-override-param-name]
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img_idx = idx // self.masks_per_image
        img = Image.open(self.img_files[img_idx]).convert("RGB")
        img = self.transform(img)

        bbox = gen_random_bbox()

        h, w = img.shape[1], img.shape[2]
        mask = bbox_to_pixel_mask(bbox, h, w, convention=self.mask_axis_convention)
        mask = mask.expand_as(img)

        masked_img = img * (1 - mask)

        return masked_img, img, bbox


def region_names(root_dir: str | Path) -> list[str]:
    """List the region identifiers (file stems) available under ``root_dir``."""
    return [p.stem for p in list_image_files(root_dir)]


def split_by_region(
    root_dir: str | Path,
    val_regions: tuple[str, ...],
    masks_per_image_train: int = 100,
    masks_per_image_val: int = 20,
    mask_axis_convention: MaskAxisConvention = "upstream",
) -> tuple[Stage1Dataset, Stage1Dataset]:
    """Build train/val :class:`Stage1Dataset` splits by held-out region.

    Every region under ``root_dir`` not in ``val_regions`` is used for
    training; ``val_regions`` (recorded explicitly in config) is the
    held-out set. Replaces the upstream ``Latent_Diffusion_Trainer`` bug of
    opening a hardcoded, nonexistent ``val_data`` directory.
    """
    all_regions = region_names(root_dir)
    missing = set(val_regions) - set(all_regions)
    if missing:
        raise ValueError(f"val_regions {sorted(missing)} not found under {root_dir}")

    train_regions = tuple(r for r in all_regions if r not in val_regions)
    if not train_regions:
        raise ValueError("val_regions covers every region; no data left for training")

    train_ds = Stage1Dataset(
        root_dir,
        masks_per_image=masks_per_image_train,
        mask_axis_convention=mask_axis_convention,
        regions=train_regions,
    )
    val_ds = Stage1Dataset(
        root_dir,
        masks_per_image=masks_per_image_val,
        mask_axis_convention=mask_axis_convention,
        regions=tuple(val_regions),
    )
    return train_ds, val_ds
