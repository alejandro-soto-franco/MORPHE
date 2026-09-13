"""Stage-2 pixel-diffusion decoder trainer.

Ported from ``Pixel_Diffusion_Decoder/trainer/cascade512_trainer.py``
(``Cascade512Trainer``). Refines a MORPHE latent (from any of the stage-1
use cases) into a full-resolution RGB image.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from accelerate import Accelerator
from diffusers import DDPMScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from morphe.datasets.dataset_cascade import PrecomputedCascadeDataset
from morphe.embeddings.interpret_cellmap import infer_cell_map
from morphe.evaluation.metrics import compute_ratio
from morphe.models.latent_adapter import LatentAdapter
from morphe.models.unet512 import UNet512


class Cascade512Trainer:
    """Trains :class:`UNet512` + :class:`LatentAdapter` to denoise a 512x512 image."""

    def __init__(
        self,
        train_index: str | Path,
        val_index: str | Path,
        checkpoint_dir: str | Path,
        pretrained_model_id: str,
        pretrained_revision: str,
        vis_dir: str | Path,
        batch_size: int = 4,
        lr: float = 1e-5,
    ):
        self.accelerator = Accelerator(mixed_precision="fp16")
        self.device = self.accelerator.device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.vis_dir = Path(vis_dir)
        self.vis_dir.mkdir(parents=True, exist_ok=True)

        self.loss_history: list[float] = []
        self.val_loss_history: list[float] = []

        self.train_loader = DataLoader(
            PrecomputedCascadeDataset(train_index), batch_size=batch_size, shuffle=True
        )
        self.val_loader = DataLoader(PrecomputedCascadeDataset(val_index), batch_size=2, shuffle=False)

        self.adapter = LatentAdapter(cz=4, cond_ch=64)
        self.unet512 = UNet512(base_ch=128, cond_ch=64, time_dim=256)

        self.optimizer = torch.optim.AdamW(
            list(self.adapter.parameters()) + list(self.unet512.parameters()),
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=1e-5,
        )

        self.scheduler2 = DDPMScheduler.from_pretrained(
            pretrained_model_id, revision=pretrained_revision, subfolder="scheduler"
        )
        self.scheduler2.config.prediction_type = "sample"

        (
            self.adapter,
            self.unet512,
            self.train_loader,
            self.val_loader,
            self.optimizer,
        ) = self.accelerator.prepare(
            self.adapter, self.unet512, self.train_loader, self.val_loader, self.optimizer
        )

    def _step(self, batch: tuple[torch.Tensor, torch.Tensor], train: bool = True) -> torch.Tensor:
        target_imgs, z_cond = batch
        z_cond = z_cond.to(self.device, dtype=torch.float16)

        cond_feats = self.adapter(z_cond)

        noise = torch.randn_like(target_imgs)
        timesteps = torch.randint(
            0, self.scheduler2.config.num_train_timesteps, (target_imgs.size(0),), device=self.device
        ).long()

        x_noisy = self.scheduler2.add_noise(target_imgs, noise, timesteps)
        x0_pred = self.unet512(x_noisy, timesteps, cond_feats)
        loss = F.mse_loss(x0_pred, target_imgs)

        if train:
            self.accelerator.backward(loss)
        return loss

    @torch.no_grad()
    def validate(self) -> float:
        self.unet512.eval()
        self.adapter.eval()
        total, n = 0.0, 0
        for batch in tqdm(self.val_loader, desc="validate"):
            total += self._step(batch, train=False).item()
            n += 1
        return total / max(1, n)

    @torch.no_grad()
    def visualize_epoch(self, epoch_idx: int, max_batches: int = 1, steps: int = 50) -> None:
        self.unet512.eval()
        self.adapter.eval()
        self.scheduler2.set_timesteps(steps, device=self.device)

        grids = []
        for i, (target_imgs, z_cond) in enumerate(self.val_loader):
            if i >= max_batches:
                break
            B = target_imgs.size(0)
            z_cond = z_cond.to(self.device, dtype=torch.float16)
            cond_feats = self.adapter(z_cond)

            x = torch.randn(B, 3, target_imgs.shape[2], target_imgs.shape[3], device=self.device)
            x = x * self.scheduler2.init_noise_sigma

            for t in self.scheduler2.timesteps:
                t_batch = torch.full((B,), int(t), device=self.device, dtype=torch.long)
                x0_pred = self.unet512(x, t_batch, cond_feats)
                x = self.scheduler2.step(x0_pred, t, x).prev_sample

            pred = (x.clamp(-1, 1) + 1) / 2
            target_vis = (target_imgs.clamp(-1, 1) + 1) / 2
            grids.append(vutils.make_grid(torch.cat([target_vis, pred], dim=0), nrow=B, padding=2))

        if grids:
            final_grid = torch.cat(grids, dim=1) if len(grids) > 1 else grids[0]
            vutils.save_image(final_grid, self.vis_dir / f"epoch_{epoch_idx:03d}.png")

    @torch.no_grad()
    def eval_composition_batch(
        self, autoencoder: torch.nn.Module, index: int, steps: int = 50, save_dir: str | Path | None = None
    ) -> None:
        """Compares the cell-type composition of a generated batch against ground truth."""
        self.unet512.eval()
        self.adapter.eval()
        save_dir = Path(save_dir) if save_dir is not None else self.vis_dir / "comp_eval"
        save_dir.mkdir(parents=True, exist_ok=True)

        self.scheduler2.set_timesteps(steps, device=self.device)
        target_imgs, z_cond = next(iter(self.val_loader))
        B = target_imgs.size(0)
        z_cond = z_cond.to(self.device, dtype=torch.float16)
        cond_feats = self.adapter(z_cond)

        x = torch.randn(B, 3, target_imgs.shape[2], target_imgs.shape[3], device=self.device)
        x = x * self.scheduler2.init_noise_sigma
        for t in self.scheduler2.timesteps:
            t_batch = torch.full((B,), int(t), device=self.device, dtype=torch.long)
            x0_pred = self.unet512(x, t_batch, cond_feats)
            x = self.scheduler2.step(x0_pred, t, x).prev_sample

        pred_imgs = (x.clamp(-1, 1) + 1) / 2
        target_vis = (target_imgs.clamp(-1, 1) + 1) / 2

        import matplotlib.pyplot as plt

        for i in range(B):
            # pyrefly: ignore [bad-argument-type]
            type_pred = infer_cell_map(pred_imgs[i], autoencoder)
            # pyrefly: ignore [bad-argument-type]
            type_orig = infer_cell_map(target_vis[i], autoencoder)
            dist_pred = compute_ratio(type_pred.squeeze().cpu().numpy())
            dist_orig = compute_ratio(type_orig.squeeze().cpu().numpy())

            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].bar(np.arange(len(dist_orig)), dist_orig)
            axes[0].set_title("original composition")
            axes[1].bar(np.arange(len(dist_pred)), dist_pred)
            axes[1].set_title("predicted composition")
            fig.tight_layout()
            fig.savefig(save_dir / f"comp_eval_{index}_img{i}.png")
            plt.close(fig)

    def save_checkpoint(self, save_dir: str | Path) -> None:
        save_dir = Path(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        self.accelerator.wait_for_everyone()
        self.accelerator.save_state(str(save_dir))

    def train(
        self,
        epochs: int = 30,
        patience: int = 5,
        vis_steps: int = 50,
        autoencoder: torch.nn.Module | None = None,
    ) -> None:
        best, bad = float("inf"), 0

        for ep in range(1, epochs + 1):
            self.unet512.train()
            self.adapter.train()
            losses = []

            for batch in tqdm(self.train_loader, desc=f"epoch {ep}"):
                with self.accelerator.accumulate(self.unet512):
                    loss = self._step(batch, train=True)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.unet512.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                losses.append(loss.item())

            train_loss = float(np.mean(losses))
            self.loss_history.append(train_loss)

            val_loss = self.validate()
            self.val_loss_history.append(val_loss)
            print(f"epoch {ep}: train={train_loss:.4f} val={val_loss:.4f}")

            self.visualize_epoch(ep, max_batches=2, steps=vis_steps)
            if autoencoder is not None:
                self.eval_composition_batch(autoencoder, ep, steps=vis_steps)

            if val_loss < best - 1e-4:
                best, bad = val_loss, 0
                self.save_checkpoint(self.checkpoint_dir)
            else:
                bad += 1
                if bad >= patience:
                    print("early stopping triggered.")
                    break
