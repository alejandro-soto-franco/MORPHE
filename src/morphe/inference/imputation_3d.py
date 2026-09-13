"""3D-imputation (z-stack slice interpolation) inference.

Ported from ``Latent_Diffusion_Generator/3D-Imputation/Infer_3d-Imputation.ipynb``
(``Slice3DInferencer``).
"""

from __future__ import annotations

from pathlib import Path

import torch
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from tqdm import tqdm

from morphe.models.cond_encoder import CondEncoder


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """``[-1, 1] -> [0, 1]``."""
    return (x.clamp(-1, 1) + 1) / 2


class Slice3DInferencer:
    def __init__(
        self,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        scheduler: DDPMScheduler,
        cond_encoder: CondEncoder,
        device: torch.device | str = "cuda",
        num_inference_steps: int = 200,
    ):
        self.device = torch.device(device)
        # pyrefly: ignore [missing-attribute]
        self.vae = vae.eval().to(self.device)
        self.scaling_factor = getattr(self.vae.config, "scaling_factor", 0.18215)
        # pyrefly: ignore [missing-attribute]
        self.unet = unet.eval().to(self.device)
        self.cond_proj = cond_encoder.eval().to(self.device)
        self.scheduler = scheduler
        # pyrefly: ignore [missing-attribute]
        self.scheduler.config.prediction_type = "sample"
        self.num_inference_steps = num_inference_steps

    @torch.no_grad()
    def infer(
        self,
        img_prev: torch.Tensor,
        img_next: torch.Tensor,
        w_prev: float = 0.5,
        w_next: float = 0.5,
        latent_save_path: str | Path | None = None,
    ) -> torch.Tensor:
        """Generate the interpolated mid-slice between ``img_prev`` and ``img_next``.

        Both inputs are ``[1, 3, H, W]`` normalised to ``[-1, 1]``.
        """
        img_prev = img_prev.to(self.device)
        img_next = img_next.to(self.device)

        latent_prev = self.vae.encode(img_prev).latent_dist.sample() * self.scaling_factor
        latent_next = self.vae.encode(img_next).latent_dist.sample() * self.scaling_factor

        condition = self.cond_proj(w_prev * latent_prev + w_next * latent_next)

        # pyrefly: ignore [missing-attribute]
        self.scheduler.set_timesteps(self.num_inference_steps, device=self.device)
        latents = torch.randn_like(latent_prev)

        # pyrefly: ignore [missing-attribute]
        for t in tqdm(self.scheduler.timesteps, leave=False):
            x0_pred = self.unet(latents, t, encoder_hidden_states=condition).sample
            # pyrefly: ignore [missing-attribute]
            latents = self.scheduler.step(model_output=x0_pred, timestep=t, sample=latents).prev_sample

        latents = latents / self.scaling_factor
        if latent_save_path is not None:
            torch.save(latents.cpu(), latent_save_path)

        return self.vae.decode(latents).sample
