"""Scores the real outpainting/2D-imputation outputs against the source region.

Only :func:`morphe.evaluation.metrics.spatial_structure_score` (LPIPS,
directly on RGB) is numerically meaningful here: the other four metrics
decode RGB into cell types via
:func:`morphe.embeddings.interpret_cellmap.infer_cell_map`, which needs an
autoencoder checkpoint trained on the original per-cell classification
probabilities. That checkpoint (``newae2.pth`` in the upstream notebook) was
never published, so those four are run against a freshly initialised,
untrained autoencoder purely to confirm the ported code executes end to end
on real images; their values are reported but flagged as not meaningful. See
``parity.md`` and the README's "Differences from upstream" section.
"""

import json
from pathlib import Path

import lpips
import numpy as np
import torch
from PIL import Image

from morphe.embeddings.autoencoder import Autoencoder
from morphe.embeddings.interpret_cellmap import infer_cell_map
from morphe.evaluation import metrics as ev

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_rgb01(path: str, size: tuple[int, int]) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize(size)
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.tensor(arr).permute(2, 0, 1)


# pyrefly: ignore [unknown-name]
outpainted = Image.open(snakemake.input.outpainted).convert("RGB")
# pyrefly: ignore [unknown-name]
imputed = Image.open(snakemake.input.imputed).convert("RGB")

# The true comparison target: the same source region resized to the
# generated output's size.
true_region_path = "Assets/sample_regions/region_0.png"
true_rgb_out = load_rgb01(true_region_path, outpainted.size)
true_rgb_imp = load_rgb01("Assets/sample_regions/region_1.png", imputed.size)

# pyrefly: ignore [unknown-name]
out_rgb = load_rgb01(snakemake.input.outpainted, outpainted.size)
# pyrefly: ignore [unknown-name]
imp_rgb = load_rgb01(snakemake.input.imputed, imputed.size)

lpips_model = lpips.LPIPS(net="vgg").to(device)

results = {
    "spatial_structure_outpainting": ev.spatial_structure_score(true_rgb_out, out_rgb, lpips_model),
    "spatial_structure_imputation_2d": ev.spatial_structure_score(true_rgb_imp, imp_rgb, lpips_model),
}

# Cell-type-dependent metrics: structural check only (see module docstring).
autoencoder = Autoencoder().to(device)
autoencoder.eval()
try:
    type_true = infer_cell_map(true_rgb_out, autoencoder)
    type_gen = infer_cell_map(out_rgb, autoencoder)
    results["rgb_centroid_distance_NOT_MEANINGFUL"] = ev.rgb_centroid_distance_score(
        true_rgb_out, out_rgb, type_true, type_gen
    )
    results["cell_type_distribution_NOT_MEANINGFUL"] = ev.cell_type_distribution(type_true, type_gen)
except Exception as exc:  # pragma: no cover - diagnostic only
    # pyrefly: ignore [bad-assignment]
    results["cell_type_metrics_error"] = str(exc)

results["_note"] = (
    # pyrefly: ignore [bad-assignment]
    "rgb_centroid_distance and cell_type_distribution use an untrained autoencoder "
    "(no published checkpoint exists); only the spatial_structure (LPIPS) scores "
    "are meaningful comparisons. See parity.md."
)

# pyrefly: ignore [unknown-name]
Path(snakemake.output.metrics).parent.mkdir(parents=True, exist_ok=True)
# pyrefly: ignore [unknown-name]
Path(snakemake.output.metrics).write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
