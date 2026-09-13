"""Stable Diffusion 2.1 inpainting finetune baseline (no text conditioning).

Ported from ``Finetune/SD2/Train_SD2.py`` and ``Finetune/SD2/SD2_Outpainting.py``.
Conditions the UNet on an OpenCLIP vision embedding of the masked image
instead of text, and widens ``conv_in`` to accept
``noisy_latent || masked_latent || mask`` concatenated on the channel axis.

All module-level constants (``SD2_BASE``, ``TRAIN_ROOT``, ``SAVE_DIR``, and
the FluxFill/finetune counterparts) were hardcoded Colab paths; every
constructor here takes them as arguments instead.
"""

from __future__ import annotations

import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection


class OutpaintDataset(Dataset):
    """Random-bbox masked/target/mask pairs, matching :mod:`morphe.datasets.stage1_dataset`."""

    def __init__(self, root_dir: str | Path, img_size: int = 512, masks_per_image: int = 40):
        self.files = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.lower().endswith(("jpg", "png", "jpeg"))
        ]
        self.masks_per_image = masks_per_image
        self.tf = transforms.Compose(
            [
                transforms.RandomChoice(
                    [
                        transforms.Lambda(lambda x: x),
                        transforms.Lambda(lambda x: TF.rotate(x, 90)),
                        transforms.Lambda(lambda x: TF.rotate(x, 180)),
                        transforms.Lambda(lambda x: TF.rotate(x, 270)),
                    ]
                ),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )

    def __len__(self) -> int:
        return len(self.files) * self.masks_per_image

    def _gen_bbox(self) -> tuple[float, float, float, float]:
        ar = random.uniform(1, 33) if random.random() < 0.5 else random.uniform(0.03, 1)
        area = random.uniform(0.1, 0.33) ** 2
        w = min(math.sqrt(area * ar), 0.99)
        h = min(math.sqrt(area / ar), 0.99)
        x1 = random.uniform(0.05, 0.99 - w)
        y1 = random.uniform(0.05, 0.99 - h)
        return x1, y1, x1 + w, y1 + h

    # pyrefly: ignore [bad-override-param-name]
    def __getitem__(self, idx: int):
        img_idx = idx // self.masks_per_image
        img = Image.open(self.files[img_idx]).convert("RGB")
        img = self.tf(img)

        bbox = self._gen_bbox()
        C, H, W = img.shape
        mask = torch.zeros_like(img)
        x1, y1 = int(bbox[0] * W), int(bbox[1] * H)
        x2, y2 = int(bbox[2] * W), int(bbox[3] * H)
        mask[:, x1:x2, y1:y2] = 1.0

        masked_img = img * (1 - mask)
        return masked_img, img, mask, torch.tensor(bbox)


def adapt_unet_conv_in(unet: UNet2DConditionModel, new_in: int) -> UNet2DConditionModel:
    """Widens ``unet.conv_in`` to ``new_in`` channels, copying old weights and mean-filling the rest."""
    # pyrefly: ignore [missing-attribute]
    conv = unet.conv_in
    old_w = conv.weight.data
    out_ch, old_in, kH, kW = old_w.shape

    new_conv = nn.Conv2d(
        new_in, out_ch, conv.kernel_size, stride=conv.stride, padding=conv.padding, bias=conv.bias is not None
    )
    with torch.no_grad():
        new_w = torch.zeros((out_ch, new_in, kH, kW), dtype=old_w.dtype)
        new_w[:, :old_in] = old_w
        mean_w = old_w.mean(dim=1, keepdim=True)
        for i in range(old_in, new_in):
            new_w[:, i : i + 1] = mean_w
        new_conv.weight.copy_(new_w)
        if conv.bias is not None:
            # pyrefly: ignore [missing-attribute]
            new_conv.bias.copy_(conv.bias)

    # pyrefly: ignore [missing-attribute]
    unet.conv_in = new_conv
    return unet


class SD21InpaintTrainer:
    """Finetunes SD2.1's UNet, conditioned on an OpenCLIP vision embedding instead of text."""

    def __init__(
        self,
        train_root: str | Path,
        val_root: str | Path,
        sd2_model_id: str,
        clip_vision_model_id: str,
        save_dir: str | Path,
        batch_size: int = 1,
        lr: float = 2e-5,
        latent_mask_threshold: float = 0.5,
        mixed_precision: str = "fp16",
    ):
        self.accelerator = Accelerator(mixed_precision=mixed_precision)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.latent_mask_threshold = latent_mask_threshold

        self.unet = UNet2DConditionModel.from_pretrained(sd2_model_id, subfolder="unet")
        self.vae = AutoencoderKL.from_pretrained(sd2_model_id, subfolder="vae")
        self.scheduler = DDPMScheduler.from_pretrained(sd2_model_id, subfolder="scheduler")

        self.vision_proc = CLIPImageProcessor.from_pretrained(clip_vision_model_id)
        self.vision_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_vision_model_id)
        self.vision_encoder.requires_grad_(False)

        latent_c = self.vae.config.latent_channels
        self.unet = adapt_unet_conv_in(self.unet, latent_c + latent_c + 1)
        self.vae.requires_grad_(False)

        self.train_loader = DataLoader(OutpaintDataset(train_root), batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(OutpaintDataset(val_root, masks_per_image=10), batch_size=batch_size)

        # pyrefly: ignore [missing-attribute]
        self.optimizer = torch.optim.AdamW([p for p in self.unet.parameters() if p.requires_grad], lr=lr)

        (
            self.unet,
            self.vae,
            self.vision_encoder,
            self.optimizer,
            self.train_loader,
            self.val_loader,
        ) = self.accelerator.prepare(
            self.unet, self.vae, self.vision_encoder, self.optimizer, self.train_loader, self.val_loader
        )
        self.scaling = self.vae.config.scaling_factor

    def _mask_to_latent(self, mask: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        _, _, h, w = shape
        m = F.interpolate(mask[:, :1], (h, w), mode="nearest")
        return (m > self.latent_mask_threshold).float()

    def _vision_embeds(self, masked: torch.Tensor) -> torch.Tensor:
        imgs = ((masked + 1) * 0.5).clamp(0, 1)
        proc = self.vision_proc(images=imgs, return_tensors="pt")
        pv = proc["pixel_values"].to(self.accelerator.device)
        with torch.no_grad():
            emb = self.vision_encoder(pixel_values=pv).image_embeds
        return emb.unsqueeze(1).expand(-1, 64, -1)

    def train_step(self, batch) -> torch.Tensor:
        masked, target, mask, _ = batch
        with torch.no_grad():
            t_lat = self.vae.encode(target).latent_dist.sample() * self.scaling

        m_lat = self._mask_to_latent(mask, t_lat.shape)
        noise = torch.randn_like(t_lat)
        ts = torch.randint(
            0, self.scheduler.config.num_train_timesteps, (t_lat.size(0),), device=self.accelerator.device
        )

        noisy_masked = self.scheduler.add_noise(t_lat * m_lat, noise * m_lat, ts)
        noisy_lat = t_lat * (1 - m_lat) + noisy_masked

        with torch.no_grad():
            masked_lat = self.vae.encode(masked).latent_dist.sample() * self.scaling

        unet_input = torch.cat([noisy_lat, masked_lat, m_lat], dim=1)
        cond = self._vision_embeds(masked)

        # pyrefly: ignore [not-callable]
        out = self.unet(unet_input, ts, encoder_hidden_states=cond).sample
        return F.mse_loss(out * m_lat, noise * m_lat)

    @torch.no_grad()
    def validate(self) -> float:
        # pyrefly: ignore [missing-attribute]
        self.unet.eval()
        total, n = 0.0, 0
        for batch in self.val_loader:
            total += self.train_step(batch).item()
            n += 1
        # pyrefly: ignore [missing-attribute]
        self.unet.train()
        return total / max(1, n)

    def train(self, epochs: int, patience: int = 4) -> None:
        best, bad = float("inf"), 0
        for ep in range(epochs):
            losses = []
            for batch in tqdm(self.train_loader, desc=f"epoch {ep + 1}"):
                with self.accelerator.accumulate(self.unet):
                    loss = self.train_step(batch)
                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                losses.append(loss.item())

            print(f"train: {np.mean(losses):.4f}")
            val = self.validate()
            print(f"val: {val:.4f}")

            if val < best:
                best, bad = val, 0
                self.accelerator.save_state(str(self.save_dir))
            else:
                bad += 1
                if bad >= patience:
                    print("early stop.")
                    return
