"""Arbitrary-shape inpainting inference (DDPM reverse sampling).

Ported from
``Latent_Diffusion_Generator/Arbitrary_Inpainting/Infer_Arbitrary_Inpainting.ipynb``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from PIL import Image

from morphe.models.cond_encoder import CondEncoder
from morphe.models.mask_encoder import MaskCoordEncoder


def load_image(path: str, device: torch.device | str) -> torch.Tensor:
    """Load an RGB image, resized to 512x512 and normalised to ``[-1, 1]``."""
    img = Image.open(path).convert("RGB").resize((512, 512))
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.tensor(arr).permute(2, 0, 1)
    return (tensor * 2 - 1).unsqueeze(0).to(device)


def load_mask(path: str, device: torch.device | str) -> torch.Tensor:
    """Load a mask image (white = masked, black = keep), resized to 512x512."""
    m = Image.open(path).convert("L").resize((512, 512))
    arr = np.array(m).astype(np.float32) / 255.0
    return torch.tensor(arr).unsqueeze(0).unsqueeze(0).to(device)


class ArbitraryInpaintEngine:
    def __init__(
        self,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        scheduler: DDPMScheduler,
        cond_encoder: CondEncoder,
        mask_encoder: MaskCoordEncoder,
        device: torch.device | str = "cuda",
    ):
        self.device = device
        # pyrefly: ignore [missing-attribute]
        self.unet = unet.eval().to(device)
        # pyrefly: ignore [missing-attribute]
        self.vae = vae.eval().to(device)
        self.scheduler = scheduler
        self.cond_proj = cond_encoder.eval().to(device)
        self.coord_encoder = mask_encoder.eval().to(device)

    @torch.no_grad()
    def inpaint(self, image: torch.Tensor, mask: torch.Tensor, num_steps: int = 200) -> np.ndarray:
        """Regenerate the masked region of ``image`` (``[1,3,512,512]``, ``[-1,1]``).

        ``mask`` is ``[1,1,512,512]``, 1 where the region is masked.
        """
        latent = self.vae.encode(image).latent_dist.sample() * self.vae.config.scaling_factor
        B, C, H, W = latent.shape

        latent_mask = F.interpolate(mask, size=(H, W), mode="nearest").expand(-1, C, -1, -1)
        masked_latent = latent * (1 - latent_mask)

        # pyrefly: ignore [missing-attribute]
        self.scheduler.set_timesteps(num_steps, device=self.device)
        noisy = torch.randn_like(latent)
        x = masked_latent + noisy * latent_mask

        # pyrefly: ignore [missing-attribute]
        for t in self.scheduler.timesteps:
            cond_tokens = self.cond_proj(masked_latent)
            coord_tokens = self.coord_encoder(mask)
            condition = torch.cat([cond_tokens, coord_tokens], dim=-1)

            noise_pred = self.unet(x, t, encoder_hidden_states=condition).sample
            # pyrefly: ignore [missing-attribute]
            x = self.scheduler.step(noise_pred, t, x).prev_sample
            x = latent_mask * x + (1 - latent_mask) * masked_latent

        image_recon = self.vae.decode(x / self.vae.config.scaling_factor).sample
        image_recon = (image_recon.clamp(-1, 1) + 1) / 2
        return image_recon[0].permute(1, 2, 0).cpu().numpy()
