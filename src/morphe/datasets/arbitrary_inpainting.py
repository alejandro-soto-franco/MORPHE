"""Irregular-polygon-mask dataset for arbitrary inpainting.

Ported from ``Latent_Diffusion_Generator/Arbitrary_Inpainting/Train_Arbitrary_Inpainting.ipynb``
(``OutpaintDataset``). Renamed :class:`ArbitraryInpaintDataset` to avoid
colliding with :class:`morphe.datasets.stage1_dataset.Stage1Dataset`'s
bbox-based masking, which this does not use.

The upstream trainer instantiated train and val datasets from the *same*
directory (only the mask count per image differed), so train and val shared
every image. :func:`split_by_region` applies the same explicit,
region-held-out split used for the outpainting/2D-imputation stage.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

from morphe.datasets.stage1_dataset import list_image_files, region_names


def gen_random_irregular_mask(
    H: int, W: int, min_area: float = 0.01, max_area: float = 0.1, num_vertices: int = 10
) -> torch.Tensor:
    """Sample a filled, star-shaped polygon mask of a random size and position."""
    cx = random.uniform(0.2, 0.8) * W
    cy = random.uniform(0.2, 0.8) * H

    target_area = random.uniform(min_area, max_area) * H * W
    theta0 = random.uniform(0, 2 * np.pi)
    angles = np.sort(np.random.uniform(0, 2 * np.pi, num_vertices))
    avg_r = np.sqrt(target_area / (math.pi * 0.3))

    radii = []
    for a in angles:
        diff = abs(a - theta0)
        diff = min(diff, 2 * np.pi - diff)
        r = avg_r * (0.7 + 0.6 * np.exp(-3.0 * diff))
        r *= 0.8 + 0.4 * np.random.rand()
        radii.append(r)

    points = [
        [int(np.clip(cx + r * np.cos(a), 0, W - 1)), int(np.clip(cy + r * np.sin(a), 0, H - 1))]
        for a, r in zip(angles, radii, strict=True)
    ]
    polygon = np.array(points)
    mask = np.zeros((H, W), dtype=np.uint8)

    for y in range(H):
        xs = []
        for i in range(num_vertices):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i + 1) % num_vertices]
            if y1 == y2:
                continue
            if min(y1, y2) <= y < max(y1, y2):
                xs.append(int(x1 + (y - y1) * (x2 - x1) / (y2 - y1)))
        xs.sort()
        for i in range(0, len(xs), 2):
            if i + 1 < len(xs):
                mask[y, xs[i] : xs[i + 1]] = 1

    return torch.tensor(mask).float().unsqueeze(0)


class ArbitraryInpaintDataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        masks_per_image: int = 50,
        regions: tuple[str, ...] | None = None,
    ):
        self.img_files = list_image_files(root_dir)
        if regions is not None:
            wanted = set(regions)
            self.img_files = [p for p in self.img_files if p.stem in wanted]
        if not self.img_files:
            raise ValueError(f"no image files found under {root_dir} (regions={regions})")
        self.masks_per_image = masks_per_image

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
        _, H, W = img.shape
        mask = gen_random_irregular_mask(H, W)
        masked_img = img * (1 - mask)
        return masked_img, img, mask


def split_by_region(
    root_dir: str | Path,
    val_regions: tuple[str, ...],
    masks_per_image_train: int = 20,
    masks_per_image_val: int = 5,
) -> tuple[ArbitraryInpaintDataset, ArbitraryInpaintDataset]:
    all_regions = region_names(root_dir)
    missing = set(val_regions) - set(all_regions)
    if missing:
        raise ValueError(f"val_regions {sorted(missing)} not found under {root_dir}")

    train_regions = tuple(r for r in all_regions if r not in val_regions)
    train_ds = ArbitraryInpaintDataset(root_dir, masks_per_image=masks_per_image_train, regions=train_regions)
    val_ds = ArbitraryInpaintDataset(
        root_dir, masks_per_image=masks_per_image_val, regions=tuple(val_regions)
    )
    return train_ds, val_ds
