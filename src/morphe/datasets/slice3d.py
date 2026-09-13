"""Z-stack slice-interpolation dataset for 3D imputation.

Ported from ``Latent_Diffusion_Generator/3D-Imputation/Train_3d-Imputation.ipynb``.
Builds ``(prev, next, mid, w_prev, w_next)`` samples from a directory of
``<z>.png`` slices: every pair of endpoints up to ``max_gap`` apart, against
every slice strictly between them, weighted by inverse distance.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

_SLICE_PATTERN = re.compile(r"^(\d+)\.png$")


class Slice3DDataset(Dataset):
    def __init__(self, root_dir: str | Path, max_gap: int = 5):
        self.root_dir = Path(root_dir)
        self.max_gap = max_gap

        self.to_tensor = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
        )

        slices = []
        for fname in os.listdir(self.root_dir):
            m = _SLICE_PATTERN.match(fname)
            if m is None:
                continue
            slices.append((int(m.group(1)), self.root_dir / fname))

        if len(slices) < 3:
            raise RuntimeError("need at least 3 slices to build samples")

        slices.sort(key=lambda x: x[0])
        n = len(slices)

        self.base_samples: list[tuple[Path, Path, Path, float, float]] = []
        for i in range(n - 2):
            z_i, path_i = slices[i]
            for j in range(i + 2, min(i + self.max_gap + 1, n)):
                z_j, path_j = slices[j]
                denom = z_j - z_i
                if denom <= 0:
                    continue
                for t in range(i + 1, j):
                    z_t, path_t = slices[t]
                    d_prev, d_next = z_t - z_i, z_j - z_t
                    w_prev = d_next / (d_prev + d_next)
                    w_next = d_prev / (d_prev + d_next)
                    self.base_samples.append((path_i, path_j, path_t, float(w_prev), float(w_next)))

        if not self.base_samples:
            raise RuntimeError("no valid interpolation samples constructed")

        self.num_rotations = 4
        self.num_hflip = 2
        self.num_vflip = 2
        self.num_aug = self.num_rotations * self.num_hflip * self.num_vflip

    def __len__(self) -> int:
        return len(self.base_samples) * self.num_aug

    # pyrefly: ignore [bad-override-param-name]
    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        base_idx = idx // self.num_aug
        rem = idx % self.num_aug

        rot_k = rem // (self.num_hflip * self.num_vflip)
        rem %= self.num_hflip * self.num_vflip
        hflip = rem // self.num_vflip
        vflip = rem % self.num_vflip

        prev_path, next_path, gt_path, w_prev, w_next = self.base_samples[base_idx]

        img_prev = Image.open(prev_path).convert("RGB")
        img_next = Image.open(next_path).convert("RGB")
        img_gt = Image.open(gt_path).convert("RGB")

        if rot_k > 0:
            angle = 90 * rot_k
            img_prev, img_next, img_gt = (
                img_prev.rotate(angle),
                img_next.rotate(angle),
                img_gt.rotate(angle),
            )
        if hflip:
            # pyrefly: ignore [missing-attribute]
            img_prev = img_prev.transpose(Image.FLIP_LEFT_RIGHT)
            # pyrefly: ignore [missing-attribute]
            img_next = img_next.transpose(Image.FLIP_LEFT_RIGHT)
            # pyrefly: ignore [missing-attribute]
            img_gt = img_gt.transpose(Image.FLIP_LEFT_RIGHT)
        if vflip:
            # pyrefly: ignore [missing-attribute]
            img_prev = img_prev.transpose(Image.FLIP_TOP_BOTTOM)
            # pyrefly: ignore [missing-attribute]
            img_next = img_next.transpose(Image.FLIP_TOP_BOTTOM)
            # pyrefly: ignore [missing-attribute]
            img_gt = img_gt.transpose(Image.FLIP_TOP_BOTTOM)

        return (
            self.to_tensor(img_prev),
            self.to_tensor(img_next),
            self.to_tensor(img_gt),
            torch.tensor(w_prev, dtype=torch.float32),
            torch.tensor(w_next, dtype=torch.float32),
        )
