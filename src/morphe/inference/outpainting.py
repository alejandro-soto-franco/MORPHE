"""Iterative outpainting inference.

Ported from
``Latent_Diffusion_Generator/Outpainting&2d-Imputation/Inferences/Inference_Outpainting.ipynb``
(``OutpaintEngine``). The duplicated ``CondEncoder``/``Transpose``/etc. classes in
that notebook are dropped in favour of the single shared implementation in
:mod:`morphe.models.cond_encoder`, which is architecturally identical and is
what the published checkpoint was actually trained with.

Preserves the upstream VAE choice (``stabilityai/sd-vae-ft-ema``, distinct
from the SD1.5-bundled VAE used at training time): this is an observed
upstream train/inference inconsistency, not one of the five confirmed bugs,
so it is kept rather than silently corrected.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from PIL import Image
from torchvision import transforms

from morphe.masks import MaskAxisConvention, bbox_to_pixel_mask, create_latent_mask
from morphe.models.cond_encoder import CondEncoder
from morphe.models.coord_encoder import CoordEncoder

Direction = Literal["right", "left", "down", "up"]
_DIRECTIONS: tuple[Direction, ...] = ("right", "left", "down", "up")


class OutpaintEngine:
    """Iteratively extends an image beyond its original field of view."""

    def __init__(
        self,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        scheduler: DDPMScheduler,
        coord_encoder: CoordEncoder,
        cond_encoder: CondEncoder,
        device: torch.device | str = "cuda",
        mask_axis_convention: MaskAxisConvention = "upstream",
    ):
        self.device = device
        # pyrefly: ignore [missing-attribute]
        self.unet = unet.eval().to(device)
        # pyrefly: ignore [missing-attribute]
        self.vae = vae.eval().to(device)
        self.scheduler = scheduler
        self.coord_encoder = coord_encoder.eval().to(device)
        self.cond_encoder = cond_encoder.eval().to(device)
        self.mask_axis_convention = mask_axis_convention

        self.transform = transforms.Compose(
            [
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )

    def _extract_old_region(
        self, image_tensor: torch.Tensor, direction_idx: int, crop_ratio: float
    ) -> torch.Tensor:
        _, _, h, w = image_tensor.shape
        masked = image_tensor.clone()

        if direction_idx in (0, 1):
            crop_w = int(w * crop_ratio)
            masked = masked[..., :, :, :crop_w] if direction_idx == 0 else masked[..., :, :, w - crop_w :]
        else:
            crop_h = int(h * crop_ratio)
            masked = masked[..., :crop_h, :] if direction_idx == 2 else masked[..., h - crop_h :, :]
        return masked

    @staticmethod
    def _extract_new_region(generated: torch.Tensor, direction_idx: int, mask_size: int) -> torch.Tensor:
        if direction_idx == 0:
            return generated[..., :, :, -mask_size:]
        if direction_idx == 1:
            return generated[..., :, :, :mask_size]
        if direction_idx == 2:
            return generated[..., :, -mask_size:, :]
        return generated[..., :, :mask_size, :]

    @staticmethod
    def _stitch_image(combined: torch.Tensor, patch: torch.Tensor, direction_idx: int) -> torch.Tensor:
        if direction_idx == 0:
            return torch.cat([combined, patch], dim=-1)
        if direction_idx == 1:
            return torch.cat([patch, combined], dim=-1)
        if direction_idx == 2:
            return torch.cat([combined, patch], dim=-2)
        return torch.cat([patch, combined], dim=-2)

    @staticmethod
    def _cyclic_shift(generated: torch.Tensor, direction_idx: int, mask_size: int) -> torch.Tensor:
        if direction_idx == 0:
            return torch.cat([generated[..., :, :, mask_size:], generated[..., :, :, :mask_size]], dim=-1)
        if direction_idx == 1:
            return torch.cat([generated[..., :, :, -mask_size:], generated[..., :, :, :-mask_size]], dim=-1)
        if direction_idx == 2:
            return torch.cat([generated[..., :, mask_size:, :], generated[..., :, :mask_size, :]], dim=-2)
        return torch.cat([generated[..., :, -mask_size:, :], generated[..., :, :-mask_size, :]], dim=-2)

    @torch.no_grad()
    def generate_iterative(
        self,
        image_tensor: torch.Tensor,
        steps: int = 200,
        crop_ratio: float = 0.97,
        iterations: int = 10,
        direction: Direction = "right",
        save_dir: str | Path | None = None,
        name: str = "default",
    ) -> np.ndarray:
        """Expand ``image_tensor`` (``[1,3,H,W]``, normalised to ``[-1,1]``) in ``direction``."""
        direction_idx = _DIRECTIONS.index(direction)
        current_image = image_tensor.clone().to(self.device)
        _, _, h, w = current_image.shape

        crop_w, crop_h = int(w * crop_ratio), int(h * crop_ratio)
        mask_size = w - crop_w if direction in ("left", "right") else h - crop_h

        # Latent grid is a fixed 64x64 (512 / 8x VAE downsampling).
        lh = lw = 64
        crop_lw, crop_lh = int(lw * crop_ratio), int(lh * crop_ratio)
        latent_mask_size = lw - crop_lw if direction in ("left", "right") else lh - crop_lh

        stitched = self._extract_old_region(current_image, direction_idx, crop_ratio)
        current_latent: torch.Tensor | None = None

        if direction == "right":
            bbox = torch.tensor([[0.0, crop_ratio, 1.0, 1.0]], device=self.device)
        elif direction == "left":
            bbox = torch.tensor([[0.0, 0.0, 1.0, 1.0 - crop_ratio]], device=self.device)
        elif direction == "down":
            bbox = torch.tensor([[crop_ratio, 0.0, 1.0, 1.0]], device=self.device)
        else:
            bbox = torch.tensor([[0.0, 0.0, 1.0 - crop_ratio, 1.0]], device=self.device)

        for i in range(iterations):
            if i == 0:
                mask = torch.zeros_like(current_image)
                pixel_mask = bbox_to_pixel_mask(bbox[0], h, w, convention=self.mask_axis_convention).to(
                    self.device
                )
                mask[:] = pixel_mask.expand_as(current_image[0])
                masked_img = current_image * (1 - mask)
                masked_latents = self.vae.encode(masked_img).latent_dist.sample()
                masked_latents = masked_latents * self.vae.config.scaling_factor
            else:
                masked_latents = current_latent

            latent_mask = create_latent_mask(
                bbox,
                # pyrefly: ignore [missing-attribute]
                masked_latents.shape,
                self.device,
                convention=self.mask_axis_convention,
            )

            # pyrefly: ignore [bad-argument-type]
            noise = torch.randn_like(masked_latents)
            # pyrefly: ignore [missing-attribute]
            noisy_latents = self.scheduler.add_noise(
                # pyrefly: ignore [unsupported-operation]
                masked_latents * latent_mask,
                noise * latent_mask,
                torch.tensor(steps),
            )
            # pyrefly: ignore [unsupported-operation]
            noisy_latents = masked_latents * (1 - latent_mask) + noisy_latents * latent_mask

            # pyrefly: ignore [missing-attribute]
            self.scheduler.set_timesteps(steps)
            latent_input = noisy_latents

            condition = torch.cat(
                [
                    self.cond_encoder(masked_latents),
                    self.coord_encoder(bbox).unsqueeze(1).expand(-1, 64, -1),
                ],
                dim=-1,
            )

            # pyrefly: ignore [missing-attribute]
            for t in self.scheduler.timesteps:
                # pyrefly: ignore [unsupported-operation]
                latent_input = latent_input * latent_mask + masked_latents * (1 - latent_mask)
                noise_pred = self.unet(latent_input, t, encoder_hidden_states=condition).sample
                # pyrefly: ignore [missing-attribute]
                latent_input = self.scheduler.step(noise_pred, t, latent_input).prev_sample

            # pyrefly: ignore [unsupported-operation]
            generated_latent = masked_latents * (1 - latent_mask) + latent_input * latent_mask
            generated_img = self.vae.decode(generated_latent / self.vae.config.scaling_factor).sample

            new_patch = self._extract_new_region(generated_img, direction_idx, mask_size)
            stitched = self._stitch_image(stitched, new_patch, direction_idx)

            if save_dir is not None:
                out_dir = Path(save_dir) / name
                out_dir.mkdir(parents=True, exist_ok=True)
                torch.save(generated_latent.cpu(), out_dir / f"{i:02d}_latent.pt")
                current_img = (generated_img[0].clamp(-1, 1) * 0.5 + 0.5).cpu()
                Image.fromarray((current_img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)).save(
                    out_dir / f"{i:02d}_image.png"
                )

            current_image = self._cyclic_shift(generated_img, direction_idx, mask_size)
            current_latent = self._cyclic_shift(generated_latent, direction_idx, latent_mask_size)

        return ((stitched[0].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8)
