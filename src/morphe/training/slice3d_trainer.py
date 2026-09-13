"""3D-imputation (z-stack slice interpolation) trainer.

Ported from ``Latent_Diffusion_Generator/3D-Imputation/Train_3d-Imputation.ipynb``
(``Slice3DTrainer``). Unlike the outpainting/2D-imputation trainer this one
already took explicit ``train_dir``/``val_dir`` arguments; the pretrained id
and checkpoint directory are the parts made config-driven here.
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

from morphe.datasets.slice3d import Slice3DDataset
from morphe.models.cond_encoder import CondEncoder
from morphe.plotting import plot_loss


class Slice3DTrainer:
    def __init__(
        self,
        train_dir: str | Path,
        val_dir: str | Path,
        pretrained_model_id: str,
        pretrained_revision: str,
        checkpoint_dir: str | Path,
        batch_size: int = 8,
        val_batch_size: int = 8,
        cond_dim: int = 768,
        lr: float = 2e-5,
        mixed_precision: str = "fp16",
    ):
        self.accelerator = Accelerator(mixed_precision=mixed_precision)
        self.checkpoint_dir = Path(checkpoint_dir)

        train_ds = Slice3DDataset(train_dir)
        val_ds = Slice3DDataset(val_dir)
        self.train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_ds, batch_size=val_batch_size)

        self.vae = AutoencoderKL.from_pretrained(
            pretrained_model_id, revision=pretrained_revision, subfolder="vae"
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_id, revision=pretrained_revision, subfolder="unet"
        )
        self.cond_proj = CondEncoder(embed_dim=cond_dim)

        self.scaling_factor = getattr(self.vae.config, "scaling_factor", 0.18215)
        self.vae.requires_grad_(False)
        self.vae.eval()

        components = [self.vae, self.unet, self.cond_proj, self.train_loader, self.val_loader]
        (
            self.vae,
            self.unet,
            self.cond_proj,
            self.train_loader,
            self.val_loader,
        ) = self.accelerator.prepare(*components)

        self.optimizer = torch.optim.AdamW(
            list(self.unet.parameters()) + list(self.cond_proj.parameters()), lr=lr
        )
        self.optimizer = self.accelerator.prepare(self.optimizer)

        self.noise_scheduler = DDPMScheduler.from_pretrained(
            pretrained_model_id, revision=pretrained_revision, subfolder="scheduler"
        )
        self.noise_scheduler.config.prediction_type = "sample"

        self.loss_history: list[float] = []
        self.val_loss_history: list[float] = []

    def train_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """The UNet predicts the target (mid) latent directly (``prediction_type="sample"``)."""
        img_prev, img_next, img_mid, wp, wn = batch

        with torch.no_grad():
            latent_prev = self.vae.encode(img_prev).latent_dist.sample() * self.scaling_factor
            latent_next = self.vae.encode(img_next).latent_dist.sample() * self.scaling_factor
            latent_mid = self.vae.encode(img_mid).latent_dist.sample() * self.scaling_factor

        noise = torch.randn_like(latent_prev)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (latent_prev.shape[0],),
            device=latent_prev.device,
            dtype=torch.long,
        )

        noisy_latents = self.noise_scheduler.add_noise(latent_mid, noise, timesteps)

        wp = wp.view(-1, 1, 1, 1)
        wn = wn.view(-1, 1, 1, 1)
        condition = self.cond_proj(wp * latent_prev + wn * latent_next)

        pred = self.unet(noisy_latents, timesteps, encoder_hidden_states=condition).sample
        return F.mse_loss(pred, latent_mid)

    def validate_step(self) -> float:
        self.unet.eval()
        total_loss, n = 0.0, 0
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="validate"):
                loss = self.train_step(batch)
                batch_size = batch[0].shape[0]
                total_loss += loss.item() * batch_size
                n += batch_size
        self.unet.train()
        return total_loss / max(1, n)

    def save_checkpoint(self, save_dir: str | Path) -> None:
        save_dir = Path(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        save_file(self.unet.state_dict(), save_dir / "unet.safetensors")
        save_file(self.cond_proj.state_dict(), save_dir / "condencoder.safetensors")

    def train(self, epochs: int = 100, patience: int = 5, lr_decay_every: int = 10) -> None:
        best_val = float("inf")
        patience_cnt = 0

        for epoch in range(1, epochs + 1):
            self.unet.train()
            epoch_losses = []
            for batch in tqdm(self.train_loader, desc=f"epoch {epoch}"):
                with self.accelerator.accumulate(self.unet):
                    loss = self.train_step(batch)
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.unet.parameters(), 1.5)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    epoch_losses.append(loss.detach().cpu().item())

            epoch_mean = float(np.mean(epoch_losses))
            self.loss_history.append(epoch_mean)

            val_loss = self.validate_step()
            self.val_loss_history.append(val_loss)
            print(f"epoch {epoch} -> train {epoch_mean:.6f} val {val_loss:.6f}")

            self.save_checkpoint(self.checkpoint_dir)
            if val_loss < best_val:
                best_val = val_loss
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= patience:
                    print("early stopping.")
                    break

            if epoch % lr_decay_every == 0:
                for g in self.optimizer.param_groups:
                    g["lr"] *= 0.5

            plot_loss(self.loss_history, self.val_loss_history, self.checkpoint_dir / "loss_curve.png")
