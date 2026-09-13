import json
from pathlib import Path

import lpips
import torch

from morphe.embeddings.autoencoder import Autoencoder
from morphe.embeddings.interpret_cellmap import infer_cell_map, load_and_recover_z3d_png
from morphe.evaluation.metrics import evaluate_generated_maps

device = "cuda" if torch.cuda.is_available() else "cpu"
# pyrefly: ignore [unknown-name]
ecfg = snakemake.config["evaluation"]

autoencoder = Autoencoder().to(device)
# pyrefly: ignore [unknown-name]
autoencoder.load_state_dict(torch.load(snakemake.input.autoencoder_checkpoint, map_location=device))
autoencoder.eval()

lpips_model = lpips.LPIPS(net="vgg").to(device)


def _load_png(path):
    return load_and_recover_z3d_png(path, white_threshold=ecfg["white_threshold"]).to(device)


def _infer(img, model):
    return infer_cell_map(img, model, z_min=tuple(ecfg["z_min"]), z_max=tuple(ecfg["z_max"]))


results = evaluate_generated_maps(
    # pyrefly: ignore [unknown-name]
    true_img_path=snakemake.input.true_map,
    # pyrefly: ignore [unknown-name]
    gen_img_paths=tuple(snakemake.input.gen_maps),
    autoencoder=autoencoder,
    lpips_model=lpips_model,
    load_png=_load_png,
    infer_cell_map=_infer,
)

# pyrefly: ignore [unknown-name]
Path(snakemake.output.metrics).parent.mkdir(parents=True, exist_ok=True)
# pyrefly: ignore [unknown-name]
Path(snakemake.output.metrics).write_text(json.dumps(results, indent=2))
