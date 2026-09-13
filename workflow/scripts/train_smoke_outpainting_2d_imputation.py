"""Runs a few real training steps on tiny, synthetic, randomly-generated
region images (see morphe.training.smoke), regardless of ``config["mode"]`` --
this must stay fast even under ``config/config.yaml``, whose
``outpainting_2d_imputation.split.root`` points at the real, full-resolution
sample regions used by the (slow, GPU) real-data reproduction instead.
"""

from pathlib import Path

from morphe.training.smoke import run_train_smoke

# pyrefly: ignore [unknown-name]
cfg = snakemake.config
smoke_cfg = cfg.get("smoke", {})

losses = run_train_smoke(
    # pyrefly: ignore [unknown-name]
    region_dir=snakemake.input[0],
    steps=smoke_cfg.get("steps", 3),
    img_size=smoke_cfg.get("img_size", 64),
    mask_axis_convention=cfg.get("mask_axis_convention", "upstream"),
)
print("smoke train losses:", losses)
assert all(loss == loss and loss >= 0 for loss in losses), "non-finite loss during smoke training"

# pyrefly: ignore [unknown-name]
Path(snakemake.output[0]).parent.mkdir(parents=True, exist_ok=True)
# pyrefly: ignore [unknown-name]
Path(snakemake.output[0]).write_text(f"losses={losses}\n")
