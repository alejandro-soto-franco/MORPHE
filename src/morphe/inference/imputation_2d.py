"""Iterative 2D-imputation (gap-filling) inference.

Ported from
``Latent_Diffusion_Generator/Outpainting&2d-Imputation/Inferences/Inference_2D-Imputation.ipynb``
(``GapfillEngine``), sharing the checkpoint and model classes with
:mod:`morphe.inference.outpainting`. Unlike the outpainting engine, the
upstream code here decodes with the checkpoint's own (SD1.5-bundled) VAE
rather than ``stabilityai/sd-vae-ft-ema``; that distinction is preserved.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from PIL import Image
from torchvision import transforms

from morphe.masks import MaskAxisConvention, create_latent_mask
from morphe.models.cond_encoder import CondEncoder
from morphe.models.coord_encoder import CoordEncoder

# Central horizontal band, normalised (x1, y1, x2, y2): the full width, rows
# 0.4375-0.5625 (a 64-image-pixel band out of 512, i.e. 8 of 64 latent rows).
CENTRAL_GAP_BBOX = (0.0, 0.4375, 1.0, 0.5625)


class GapfillEngine:
    """Iteratively regenerates a central horizontal gap in latent space."""

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

    @torch.no_grad()
    def generate_iterative(
        self,
        image_tensor: torch.Tensor,
        steps: int = 200,
        iterations: int = 10,
        save_dir: str | Path | None = None,
        name: str = "default",
    ) -> np.ndarray:
        """Regenerate the central gap of ``image_tensor`` (``[1,3,H,W]``, ``[-1,1]``)."""
        current_image = image_tensor.clone().to(self.device)
        bbox = torch.tensor([list(CENTRAL_GAP_BBOX)], device=self.device)

        current_latent = self.vae.encode(current_image).latent_dist.sample()
        current_latent = current_latent * self.vae.config.scaling_factor

        preview: np.ndarray | None = None

        for i in range(iterations):
            latent_mask = create_latent_mask(
                bbox,
                # pyrefly: ignore [bad-argument-type]
                current_latent.shape,
                self.device,
                convention=self.mask_axis_convention,
            )
            masked_latent = current_latent * (1 - latent_mask)

            noise = torch.randn_like(current_latent)
            # pyrefly: ignore [missing-attribute]
            noisy_latent = self.scheduler.add_noise(
                masked_latent * latent_mask, noise * latent_mask, torch.tensor(steps)
            )
            noisy_latent = masked_latent * (1 - latent_mask) + noisy_latent * latent_mask

            # pyrefly: ignore [missing-attribute]
            self.scheduler.set_timesteps(steps)
            latent_input = noisy_latent

            condition = torch.cat(
                [
                    self.cond_encoder(masked_latent),
                    self.coord_encoder(bbox).unsqueeze(1).expand(-1, 64, -1),
                ],
                dim=-1,
            )

            # pyrefly: ignore [missing-attribute]
            for t in self.scheduler.timesteps:
                latent_input = latent_input * latent_mask + masked_latent * (1 - latent_mask)
                noise_pred = self.unet(latent_input, t, encoder_hidden_states=condition).sample
                # pyrefly: ignore [missing-attribute]
                latent_input = self.scheduler.step(noise_pred, t, latent_input).prev_sample

            if i == iterations - 1:
                generated_latent = latent_input * latent_mask + masked_latent * (1 - latent_mask)
                generated_img = self.vae.decode(generated_latent / self.vae.config.scaling_factor).sample
                preview_arr = generated_img[0].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5
                preview = (preview_arr.clip(0, 1) * 255).astype(np.uint8)
                if save_dir is not None:
                    out_dir = Path(save_dir) / name
                    out_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(generated_latent.cpu(), out_dir / f"{i}_latent.pt")
                    # pyrefly: ignore [bad-argument-type]
                    Image.fromarray(preview).save(out_dir / f"{i}_image.png")
                # pyrefly: ignore [bad-return]
                return preview

            # Splice the newly generated 8-latent-pixel-wide centre back into
            # the running latent, keeping the tensor at 64 latent columns.
            new_patch_latent = latent_input[..., :, :, 28:36]
            left_latent_patch = new_patch_latent[..., :, :, :4]
            right_latent_patch = new_patch_latent[..., :, :, 4:]

            cropped_latent = current_latent[..., :, :, 4:-4]
            cropped_left = cropped_latent[..., :, :, :24]
            cropped_right = cropped_latent[..., :, :, -24:]

            white_gap_latent = torch.zeros(
                (1, current_latent.shape[1], current_latent.shape[2], 8), device=self.device
            )

            stitched_latent = torch.cat(
                [cropped_left, left_latent_patch, white_gap_latent, right_latent_patch, cropped_right],
                dim=-1,
            )
            start = (stitched_latent.shape[-1] - 64) // 2
            current_latent = stitched_latent[..., :, :, start : start + 64]

        assert preview is not None
        return preview
