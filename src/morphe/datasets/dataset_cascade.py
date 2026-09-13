"""Precomputed (latent, target-image) pairs for the pixel-diffusion decoder.

Ported unchanged from ``Pixel_Diffusion_Decoder/data/dataset_cascade.py``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset


class PrecomputedCascadeDataset(Dataset):
    """Loads ``.pt`` files each holding ``z_cond`` (``[4,64,64]``) and ``target_img`` (``[3,512,512]``)."""

    def __init__(self, index_jsonl: str | Path):
        self.items: list[str] = []
        with open(index_jsonl) as f:
            for line in f:
                path = json.loads(line)["pt"]
                if os.path.exists(path):
                    self.items.append(path)
                else:
                    print(f"[warn] missing file: {path}")

    def __len__(self) -> int:
        return len(self.items)

    # pyrefly: ignore [bad-override-param-name]
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        pack = torch.load(self.items[idx], map_location="cpu")
        return pack["target_img"].float(), pack["z_cond"].to(torch.float16)
