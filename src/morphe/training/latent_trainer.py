"""Stage-1 latent-diffusion trainer for outpainting / 2D-imputation.

Ported from
``Latent_Diffusion_Generator/Outpainting&2d-Imputation/Train/trainers/Latent_Diffusion_Trainer.py``.

Fixes three of the confirmed upstream defects:

* ``Latent_Diffusion_Trainer.py:22`` opened a hardcoded, nonexistent
  ``val_data`` directory. :class:`LatentTrainer` now takes an explicit,
  config-driven train/val region split (see
  :func:`morphe.datasets.stage1_dataset.split_by_region`).
* The default pretrained id ``runwayml/stable-diffusion-v1-5`` was removed
  from HuggingFace. The pretrained id and revision are now config-driven
  (see :class:`morphe.config.PretrainedConfig`), defaulting to a maintained
  mirror pinned by commit hash.
* The checkpoint directory (``drive/MyDrive/checkpoint-merfish``) and the
  loss-plot output were Colab-specific hardcoded paths; both are now
  constructor arguments.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from torch.utils.data import DataLoader
from tqdm import tqdm

from morphe.datasets.stage1_dataset import Stage1Dataset
from morphe.masks import MaskAxisConvention, create_latent_mask
from morphe.models.cond_encoder import CondEncoder
from morphe.models.coord_encoder import CoordEncoder
from morphe.plotting import plot_loss


class LatentTrainer:
    """Trains the UNet + coordinate/conditioning encoders for stage-1 latent diffusion."""

    def __init__(
        self,
        train_dataset: Stage1Dataset,
        val_dataset: Stage1Dataset,
        checkpoint_dir: str | Path,
        pretrained_model_id: str,
        pretrained_revision: str,
        mask_axis_convention: MaskAxisConvention = "upstream",
        train_batch_size: int = 16,
        val_batch_size: int = 4,
        lr: float = 2e-5,
        mixed_precision: str = "fp16",
    ):
        self.accelerator = Accelerator(mixed_precision=mixed_precision)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.mask_axis_convention = mask_axis_convention

        self.train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=val_batch_size)

        self.vae = AutoencoderKL.from_pretrained(
            pretrained_model_id, revision=pretrained_revision, subfolder="vae"
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_id, revision=pretrained_revision, subfolder="unet"
        )

        self.coord_enc = CoordEncoder()
        self.cond_enc = CondEncoder()

        self.optimizer = torch.optim.AdamW(
            list(self.unet.parameters())
            + list(self.coord_enc.parameters())
            + list(self.cond_enc.parameters()),
            lr=lr,
        )

        components = [
            self.vae,
            self.unet,
            self.coord_enc,
            self.cond_enc,
            self.train_loader,
            self.val_loader,
            self.optimizer,
        ]
        (
            self.vae,
            self.unet,
            self.coord_enc,
            self.cond_enc,
            self.train_loader,
            self.val_loader,
            self.optimizer,
        ) = self.accelerator.prepare(*components)

        self.vae.requires_grad_(False)
        self.scheduler = DDPMScheduler.from_pretrained(
            pretrained_model_id, revision=pretrained_revision, subfolder="scheduler"
        )

        self.train_loss: list[float] = []
        self.val_loss: list[float] = []

    def train_step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        masked_img, target_img, bbox = batch

        with torch.no_grad():
            target_lat = self.vae.encode(target_img).latent_dist.sample()
            target_lat = target_lat * self.vae.config.scaling_factor

        mask = create_latent_mask(
            bbox, target_lat.shape, target_lat.device, convention=self.mask_axis_convention
        )

        noise = torch.randn_like(target_lat)
        t = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (target_lat.size(0),),
            device=target_lat.device,
        )

        noisy = self.scheduler.add_noise(target_lat * mask, noise * mask, t)
        noisy = target_lat * (1 - mask) + noisy

        with torch.no_grad():
            masked_lat = self.vae.encode(masked_img).latent_dist.sample()
            masked_lat = masked_lat * self.vae.config.scaling_factor

        cond_bbox = self.coord_enc(bbox)
        cond_tokens = torch.cat(
            [self.cond_enc(masked_lat), cond_bbox.unsqueeze(1).expand(-1, 64, -1)], dim=-1
        )

        pred = self.unet(noisy, t, encoder_hidden_states=cond_tokens).sample

        return F.mse_loss(pred * mask, noise * mask)

    def validate(self) -> float:
        self.unet.eval()
        total = 0.0
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="validate"):
                total += self.train_step(batch).item()
        return total / len(self.val_loader)

    def train(self, epochs: int = 10, patience: int = 5) -> None:
        best_val = float("inf")
        bad_epochs = 0

        for ep in range(epochs):
            self.unet.train()
            ep_losses = []

            for batch in tqdm(self.train_loader, desc=f"epoch {ep}"):
                with self.accelerator.accumulate():
                    loss = self.train_step(batch)
                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                ep_losses.append(loss.item())

            train_avg = sum(ep_losses) / len(ep_losses)
            val_avg = self.validate()

            self.train_loss.append(train_avg)
            self.val_loss.append(val_avg)

            print(f"epoch {ep}: train={train_avg:.4f} val={val_avg:.4f}")

            if val_avg < best_val:
                best_val = val_avg
                bad_epochs = 0
                self.accelerator.save_state(str(self.checkpoint_dir))
            else:
                bad_epochs += 1

            plot_loss(self.train_loss, self.val_loss, self.checkpoint_dir / "loss_curve.png")

            if bad_epochs >= patience:
                print("early stopping.")
                break
