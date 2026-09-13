from pathlib import Path

import torch
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from PIL import Image

from morphe.inference.checkpoints import load_stage1_components
from morphe.inference.outpainting import OutpaintEngine
from morphe.models.cond_encoder import CondEncoder
from morphe.models.coord_encoder import CoordEncoder

# pyrefly: ignore [unknown-name]
cfg = snakemake.config
pretrained = cfg["pretrained"]
outpainting_vae_cfg = cfg["outpainting_inference_vae"]
oc = cfg["outpainting_2d_imputation"]

device = "cuda" if torch.cuda.is_available() else "cpu"

unet = UNet2DConditionModel.from_pretrained(
    pretrained["model_id"], revision=pretrained["revision"], subfolder="unet"
)
vae = AutoencoderKL.from_pretrained(outpainting_vae_cfg["model_id"], revision=outpainting_vae_cfg["revision"])
scheduler = DDPMScheduler.from_pretrained(
    pretrained["model_id"], revision=pretrained["revision"], subfolder="scheduler"
)
coord_encoder = CoordEncoder()
cond_encoder = CondEncoder()

# pyrefly: ignore [unknown-name]
checkpoint_dir = Path(snakemake.input.unet).parent
load_stage1_components(checkpoint_dir, unet, coord_encoder, cond_encoder, device=device)

engine = OutpaintEngine(
    unet=unet,
    vae=vae,
    scheduler=scheduler,
    coord_encoder=coord_encoder,
    cond_encoder=cond_encoder,
    device=device,
    mask_axis_convention=cfg.get("mask_axis_convention", "upstream"),
)

# pyrefly: ignore [unknown-name]
img = Image.open(snakemake.input.region).convert("RGB")
img_t = engine.transform(img).unsqueeze(0).to(device)

result = engine.generate_iterative(
    img_t,
    steps=oc["inference"]["steps"],
    crop_ratio=oc["inference"]["outpainting_crop_ratio"],
    iterations=oc["inference"]["outpainting_iterations"],
    direction="right",
)

# pyrefly: ignore [unknown-name]
Path(snakemake.output.image).parent.mkdir(parents=True, exist_ok=True)
# pyrefly: ignore [unknown-name]
Image.fromarray(result).save(snakemake.output.image)
