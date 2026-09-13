"""A from-scratch, tiny-resolution training smoke test.

``LatentTrainer``, ``Slice3DTrainer``, ``ArbitraryInpaintTrainer`` and
``Cascade512Trainer`` all call ``from_pretrained`` on a full-size Stable
Diffusion checkpoint, which neither fits the 30-minute compute cap nor runs
without a multi-gigabyte download. This module builds tiny, randomly
initialised ``UNet2DConditionModel``/``AutoencoderKL`` configs directly (no
download) and runs a handful of real training steps through the same
:func:`morphe.masks.create_latent_mask` / :class:`morphe.models.cond_encoder.CondEncoder`
/ :class:`morphe.models.coord_encoder.CoordEncoder` machinery the production
trainer uses, at a resolution small enough to finish on CPU in seconds.

This exercises the real training-step logic (masking, noise scheduling,
conditioning, loss), not a placeholder.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel

from morphe.datasets.stage1_dataset import Stage1Dataset
from morphe.masks import MaskAxisConvention, create_latent_mask
from morphe.models.cond_encoder import CondEncoder
from morphe.models.coord_encoder import CoordEncoder


def build_tiny_vae(latent_channels: int = 4) -> AutoencoderKL:
    return AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",),
        up_block_types=("UpDecoderBlock2D",),
        block_out_channels=(8,),
        layers_per_block=1,
        latent_channels=latent_channels,
        norm_num_groups=4,
    )


def build_tiny_unet(latent_channels: int = 4, cross_attention_dim: int = 32) -> UNet2DConditionModel:
    return UNet2DConditionModel(
        sample_size=8,
        in_channels=latent_channels,
        out_channels=latent_channels,
        layers_per_block=1,
        block_out_channels=(16, 32),
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        cross_attention_dim=cross_attention_dim,
        attention_head_dim=4,
        norm_num_groups=4,
    )


def run_train_smoke(
    region_dir: str | Path,
    steps: int = 3,
    img_size: int = 64,
    mask_axis_convention: MaskAxisConvention = "upstream",
    seed: int = 0,
) -> list[float]:
    """Runs ``steps`` real (masking + noise + UNet + loss) training iterations.

    Returns the per-step loss, mostly so a test can assert it is finite.
    """
    torch.manual_seed(seed)

    cond_dim = 24
    embed_dim = 16
    latent_channels = 4

    vae = build_tiny_vae(latent_channels)
    unet = build_tiny_unet(latent_channels, cross_attention_dim=embed_dim + cond_dim)
    coord_enc = CoordEncoder(dim=cond_dim)
    cond_enc = CondEncoder(in_channels=latent_channels, embed_dim=embed_dim, num_tokens=1)
    scheduler = DDPMScheduler(num_train_timesteps=50)

    dataset = Stage1Dataset(
        region_dir, img_size=img_size, masks_per_image=steps, mask_axis_convention=mask_axis_convention
    )

    optimizer = torch.optim.AdamW(
        # pyrefly: ignore [missing-attribute]
        list(unet.parameters()) + list(coord_enc.parameters()) + list(cond_enc.parameters()),
        lr=1e-3,
    )

    losses = []
    for i in range(steps):
        masked_img, target_img, bbox = dataset[i]
        masked_img = masked_img.unsqueeze(0)
        target_img = target_img.unsqueeze(0)
        bbox = bbox.unsqueeze(0)

        with torch.no_grad():
            # pyrefly: ignore [missing-attribute]
            target_lat = vae.encode(target_img).latent_dist.sample() * vae.config.scaling_factor
            # pyrefly: ignore [missing-attribute]
            masked_lat = vae.encode(masked_img).latent_dist.sample() * vae.config.scaling_factor

        mask = create_latent_mask(bbox, target_lat.shape, target_lat.device, convention=mask_axis_convention)

        noise = torch.randn_like(target_lat)
        # pyrefly: ignore [missing-attribute]
        t = torch.randint(0, scheduler.config.num_train_timesteps, (1,))
        # pyrefly: ignore [missing-attribute]
        noisy = scheduler.add_noise(target_lat * mask, noise * mask, t)
        noisy = target_lat * (1 - mask) + noisy

        cond_bbox = coord_enc(bbox)
        num_tokens = cond_enc(masked_lat).shape[1]
        cond_tokens = torch.cat(
            [cond_enc(masked_lat), cond_bbox.unsqueeze(1).expand(-1, num_tokens, -1)], dim=-1
        )

        # pyrefly: ignore [not-callable]
        pred = unet(noisy, t, encoder_hidden_states=cond_tokens).sample
        loss = F.mse_loss(pred * mask, noise * mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return losses
