"""Synthetic region-image fixtures for smoke runs and tests.

Shaped like the real region images (region-code stems the loaders' file
filter accepts) but generated on the fly, so ``pixi run smoke`` and CI need
no data checked into or downloaded into the repository.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def make_synthetic_regions(
    out_dir: str | Path, n_regions: int = 6, size: int = 64, seed: int = 0
) -> list[Path]:
    """Writes ``n_regions`` random RGB PNGs named ``region_<i>.png`` into ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    paths = []
    for i in range(n_regions):
        arr = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
        path = out_dir / f"region_{i}.png"
        Image.fromarray(arr).save(path)
        paths.append(path)
    return paths
