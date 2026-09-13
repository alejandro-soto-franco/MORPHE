"""Arbitrary-shape inpainting trainer.

Ported from
``Latent_Diffusion_Generator/Arbitrary_Inpainting/Train_Arbitrary_Inpainting.ipynb``
(``OutpaintTrainer_new``).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from tqdm import tqdm

from morphe.datasets.arbitrary_inpainting import ArbitraryInpaintDataset
from morphe.models.cond_encoder import CondEncoder
from morphe.models.mask_encoder import MaskCoordEncoder
from morphe.plotting import plot_loss


class ArbitraryInpaintTrainer:
    def __init__(
        self,
        train_dataset: ArbitraryInpaintDataset,
        val_dataset: ArbitraryInpaintDataset,
        checkpoint_dir: str | Path,
        pretrained_model_id: str,
        pretrained_revision: str,
        train_batch_size: int = 16,
        val_batch_size: int = 4,
        lr: float = 2e-5,
        mixed_precision: str = "fp16",
    ):
        self.accelerator = Accelerator(mixed_precision=mixed_precision)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.loss_history: list[float] = []
        self.val_loss_history: list[float] = []

        self.train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=val_batch_size)

        self.vae = AutoencoderKL.from_pretrained(
            pretrained_model_id, revision=pretrained_revision, subfolder="vae"
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_id, revision=pretrained_revision, subfolder="unet"
        )
        latent_c = self.vae.config.latent_channels

        self.cond_proj = CondEncoder(in_channels=latent_c, embed_dim=736)
        self.coord_encoder = MaskCoordEncoder(embed_dim=32, num_tokens=64)

        components = [
            self.vae,
            self.unet,
            self.cond_proj,
            self.coord_encoder,
            self.train_loader,
            self.val_loader,
        ]
        (
            self.vae,
            self.unet,
            self.cond_proj,
            self.coord_encoder,
            self.train_loader,
            self.val_loader,
        ) = self.accelerator.prepare(*components)

        self.optimizer = torch.optim.AdamW(
            list(self.unet.parameters())
            + list(self.cond_proj.parameters())
            + list(self.coord_encoder.parameters()),
            lr=lr,
        )
        self.optimizer = self.accelerator.prepare(self.optimizer)

        self.noise_scheduler = DDPMScheduler.from_pretrained(
            pretrained_model_id, revision=pretrained_revision, subfolder="scheduler"
        )
        self.vae.requires_grad_(False)

    def save_checkpoint(self, save_dir: str | Path) -> None:
        save_dir = Path(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        save_file(self.unet.state_dict(), save_dir / "unet.safetensors")
        save_file(self.cond_proj.state_dict(), save_dir / "condencoder.safetensors")
        save_file(self.coord_encoder.state_dict(), save_dir / "coordencoder.safetensors")

    def train_step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        masked_imgs, target_imgs, mask = batch

        with torch.no_grad():
            target_latents = self.vae.encode(target_imgs).latent_dist.sample()
            target_latents = target_latents * self.vae.config.scaling_factor

        B, C, lh, lw = target_latents.shape
        latent_mask = torch.nn.functional.interpolate(mask, size=(lh, lw), mode="nearest")
        latent_mask = latent_mask.expand(-1, C, -1, -1)

        noise = torch.randn_like(target_latents)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (B,), device=target_latents.device
        )

        noisy_latents = self.noise_scheduler.add_noise(
            target_latents * latent_mask, noise * latent_mask, timesteps
        )
        noisy_latents = target_latents * (1 - latent_mask) + noisy_latents

        with torch.no_grad():
            masked_latents = self.vae.encode(masked_imgs).latent_dist.sample()
            masked_latents = masked_latents * self.vae.config.scaling_factor

        cond_tokens = self.cond_proj(masked_latents)
        coord_tokens = self.coord_encoder(mask)
        condition = torch.cat([cond_tokens, coord_tokens], dim=-1)

        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states=condition).sample
        return F.mse_loss(noise_pred * latent_mask, noise * latent_mask)

    def validate_step(self) -> float:
        self.unet.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for batch in self.val_loader:
                total += self.train_step(batch).item()
                count += 1
        return total / max(count, 1)

    def train(self, epochs: int = 20) -> None:
        best_val_loss = float("inf")
        for ep in range(epochs):
            self.unet.train()
            losses = []
            for batch in tqdm(self.train_loader, desc=f"epoch {ep}"):
                with self.accelerator.accumulate(self.unet):
                    loss = self.train_step(batch)
                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                losses.append(loss.item())

            train_loss = float(np.mean(losses))
            self.loss_history.append(train_loss)

            val_loss = self.validate_step()
            self.val_loss_history.append(val_loss)

            print(f"epoch {ep}: train={train_loss:.4f} val={val_loss:.4f}")
            plot_loss(self.loss_history, self.val_loss_history, self.checkpoint_dir / "loss_curve.png")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(self.checkpoint_dir)
