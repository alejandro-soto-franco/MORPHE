"""Precomputing decoder training pairs, and decoding a MORPHE latent to a full image.

Ported from ``Pixel_Diffusion_Decoder/Dataset_Construct_for_Decoder.ipynb``
and ``Pixel_Diffusion_Decoder/Infer_Decoder.ipynb``.

Upstream defect (not one of the five confirmed bugs, found while porting):
``precompute_latents`` called ``OutpaintDataset(..., no_mask=True)``, a
keyword argument neither :class:`morphe.datasets.stage1_dataset.Stage1Dataset`
nor :class:`morphe.datasets.arbitrary_inpainting.ArbitraryInpaintDataset`
accept, so the cell cannot run as shipped. :func:`precompute_latents` here
takes an already-constructed dataset (of any of the stage-1 use cases)
yielding ``(augmented_img, target_img)`` pairs instead.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from diffusers import AutoencoderKL, DDPMScheduler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from morphe.models.latent_adapter import LatentAdapter
from morphe.models.unet512 import UNet512


class VAEEncoder(torch.nn.Module):
    """Wraps a frozen Stable-Diffusion VAE's ``encode`` as ``imgs -> scaled latents``."""

    def __init__(self, pretrained_model_id: str, pretrained_revision: str, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.vae = AutoencoderKL.from_pretrained(
            pretrained_model_id, revision=pretrained_revision, subfolder="vae"
        ).to(device)
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)
        self.scaling = self.vae.config.scaling_factor

    @torch.no_grad()
    def encode(self, imgs: torch.Tensor) -> torch.Tensor:
        """``imgs``: ``[B,3,512,512]`` in ``[-1,1]``. Returns ``[B,4,64,64]``."""
        return self.vae.encode(imgs).latent_dist.sample() * self.scaling


def precompute_latents(
    dataset: Dataset,
    out_dir: str | Path,
    encoder: VAEEncoder,
    batch_size: int = 16,
) -> Path:
    """Encode every ``(augmented_img, target_img)`` pair in ``dataset`` and index it.

    Writes one ``.pt`` file per sample (``z_cond``, ``target_img``, both
    fp16) plus an ``index.jsonl`` consumed by
    :class:`morphe.datasets.dataset_cascade.PrecomputedCascadeDataset`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.jsonl"

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    with open(index_path, "w") as fw:
        global_idx = 0
        for aug_img, target_img in tqdm(loader, desc=f"precomputing to {out_dir}"):
            aug_b = aug_img.to(encoder.device)
            with torch.no_grad():
                z_cond = encoder.encode(aug_b)

            for b in range(aug_b.size(0)):
                idx = global_idx + b
                pt_path = out_dir / f"sample_{idx:06d}.pt"
                sample = {
                    "z_cond": z_cond[b].detach().cpu().half(),
                    "target_img": target_img[b].detach().cpu().half(),
                }
                torch.save(sample, pt_path)
                fw.write(json.dumps({"pt": str(pt_path)}) + "\n")
            global_idx += aug_b.size(0)

    return index_path


class DecoderInferenceEngine:
    """Decodes a stage-1 latent into a full 512x512 RGB image via cascade diffusion."""

    def __init__(
        self,
        adapter: LatentAdapter,
        unet: UNet512,
        scheduler: DDPMScheduler,
        device: str = "cuda",
        num_inference_steps: int = 150,
    ):
        self.device = device
        self.adapter = adapter.eval().to(device)
        self.unet = unet.eval().to(device)
        self.scheduler = scheduler
        # pyrefly: ignore [missing-attribute]
        self.scheduler.config.prediction_type = "sample"
        self.num_inference_steps = num_inference_steps

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """``z``: ``[4,64,64]`` or ``[1,4,64,64]`` stage-1 latent. Returns ``[1,3,512,512]`` in ``[0,1]``."""
        z = z.to(device=self.device, dtype=torch.float16)
        if z.ndim == 3:
            z = z.unsqueeze(0)

        # pyrefly: ignore [missing-attribute]
        self.scheduler.set_timesteps(self.num_inference_steps, device=self.device)
        cond_feats = self.adapter(z)

        # pyrefly: ignore [missing-attribute]
        x = torch.randn(1, 3, 512, 512, device=self.device) * self.scheduler.init_noise_sigma
        # pyrefly: ignore [missing-attribute]
        for t in self.scheduler.timesteps:
            t_batch = torch.full((1,), int(t), device=self.device, dtype=torch.long)
            x0_pred = self.unet(x, t_batch, cond_feats)
            # pyrefly: ignore [missing-attribute]
            x = self.scheduler.step(x0_pred, t, x).prev_sample

        return (x.clamp(-1, 1) + 1) / 2

    def decode_to_pil(self, z: torch.Tensor):
        out_img = self.decode(z)
        return transforms.ToPILImage()(out_img[0].float().cpu())
