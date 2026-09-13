"""FLUX.1-Fill LoRA finetune baseline (no text conditioning).

Ported from ``Finetune/FluxFill/Train_FluxFill.py``. LoRA-adapts
``black-forest-labs/FLUX.1-Fill-dev``'s transformer with zeroed text/pooled
embeddings, following the packed-latent layout ``FluxFillPipeline`` uses
internally.

Module-level constants (``HF_REPO``, ``TRAIN_ROOT``, ``SAVE_DIR``, the LoRA
target modules) were hardcoded; every constructor argument here is explicit.
This baseline is ported for training fidelity; its companion iterative
outpainting inference script (``Finetune/FluxFill/FluxFill_Outpainting.py``)
duplicates the cyclic-shift stitching already covered by
:mod:`morphe.inference.outpainting` for MORPHE's own model and is not
separately re-ported.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator

# pyrefly: ignore [missing-module-attribute]
from diffusers import FluxFillPipeline
from diffusers.training_utils import cast_training_params
from peft import LoraConfig
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm import tqdm

DEFAULT_LORA_TARGET_MODULES = [
    "attn.to_k",
    "attn.to_q",
    "attn.to_v",
    "attn.to_out.0",
    "attn.add_k_proj",
    "attn.add_q_proj",
    "attn.add_v_proj",
    "attn.to_add_out",
    "ff.net.0.proj",
    "ff.net.2",
    "ff_context.net.0.proj",
    "ff_context.net.2",
]


class RandomMaskDataset(Dataset):
    """Random-bbox masked/target/mask triples with a discrete rotation augmentation."""

    def __init__(self, img_dir: str | Path, size: int = 512, masks_per_image: int = 50):
        self.size = size
        self.masks_per_image = masks_per_image
        self.img_paths = sorted(
            str(p) for p in Path(img_dir).iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")
        )
        if not self.img_paths:
            raise ValueError(f"no images found in {img_dir}")

        self.resize = transforms.Resize((size, size))
        self.to_tensor = transforms.ToTensor()
        self.rotations = [0, 90, 180, 270]

    def _gen_bbox(self) -> tuple[float, float, float, float]:
        ar = random.uniform(1, 33) if random.random() < 0.5 else random.uniform(0.03, 1)
        area = random.uniform(0.1, 0.33) ** 2
        w = min(math.sqrt(area * ar), 0.99)
        h = min(math.sqrt(area / ar), 0.99)
        x1 = random.uniform(0.01, 0.99 - w)
        y1 = random.uniform(0.01, 0.99 - h)
        return x1, y1, x1 + w, y1 + h

    def __len__(self) -> int:
        return max(1, len(self.img_paths) * self.masks_per_image)

    # pyrefly: ignore [bad-override-param-name]
    def __getitem__(self, idx: int):
        img_idx = idx // self.masks_per_image
        img = Image.open(self.img_paths[img_idx % len(self.img_paths)]).convert("RGB")
        img = self.to_tensor(self.resize(img)) * 2 - 1

        H, W = self.size, self.size
        x1, y1, x2, y2 = self._gen_bbox()
        x1p, y1p, x2p, y2p = int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)
        x2p, y2p = max(x2p, x1p + 1), max(y2p, y1p + 1)

        mask = torch.zeros(1, H, W)
        mask[:, x1p:x2p, y1p:y2p] = 1.0
        masked_img = img * (1 - mask)

        angle = random.choice(self.rotations)
        img = TF.rotate(img, angle, interpolation=TF.InterpolationMode.NEAREST)
        masked_img = TF.rotate(masked_img, angle, interpolation=TF.InterpolationMode.NEAREST)

        return img, mask, masked_img


def _pack_mask(mask: torch.Tensor, lat: torch.Tensor, vae_scale: int) -> torch.Tensor:
    """Reshapes a full-resolution mask into the packed-latent-grid layout FLUX expects."""
    B, C, Hlat, Wlat = lat.shape
    m = mask.reshape(-1, 1, Hlat * vae_scale, Wlat * vae_scale)[:, 0, :, :]
    m = m.view(B, Hlat, vae_scale, Wlat, vae_scale)
    m = m.permute(0, 2, 4, 1, 3)
    return m.reshape(B, vae_scale * vae_scale, Hlat, Wlat)


class FluxFillLoraTrainer:
    def __init__(
        self,
        train_root: str | Path,
        val_root: str | Path,
        hf_repo: str,
        save_dir: str | Path,
        lora_rank: int = 8,
        img_size: int = 512,
        batch_size: int = 1,
        lr: float = 1e-5,
        mixed_precision: str = "fp16",
    ):
        self.acc = Accelerator(mixed_precision=mixed_precision)
        self.device = self.acc.device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.pipe = FluxFillPipeline.from_pretrained(hf_repo, torch_dtype=torch.float16)
        self.vae = self.pipe.vae.to(self.device)
        self.transformer = self.pipe.transformer
        self.scheduler = self.pipe.scheduler

        for k, v in list(self.scheduler.__dict__.items()):
            if isinstance(v, torch.Tensor):
                setattr(self.scheduler, k, v.to(self.device))

        self.transformer.requires_grad_(False)
        self.vae.requires_grad_(False)

        lora_cfg = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank,
            init_lora_weights="gaussian",
            target_modules=DEFAULT_LORA_TARGET_MODULES,
        )
        self.transformer.add_adapter(lora_cfg)
        cast_training_params([self.transformer], dtype=torch.float32)

        self.ds = RandomMaskDataset(train_root, size=img_size)
        self.dl = DataLoader(self.ds, batch_size=batch_size, shuffle=True)
        self.val_ds = RandomMaskDataset(val_root, size=img_size, masks_per_image=10)
        self.val_dl = DataLoader(self.val_ds, batch_size=1, shuffle=False)

        trainable = [p for p in self.transformer.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(trainable, lr=lr)

        self.transformer, self.opt, self.dl = self.acc.prepare(self.transformer, self.opt, self.dl)
        self.vae.eval()

    def _step(self, batch, train: bool) -> torch.Tensor:
        img, mask, masked_img = (t.to(self.device, dtype=self.vae.dtype) for t in batch)
        vae_scale = 2 ** (len(self.vae.config.block_out_channels) - 1)

        shift, scale = self.vae.config.shift_factor, self.vae.config.scaling_factor
        with torch.no_grad():
            lat = (self.vae.encode(img).latent_dist.sample() - shift) * scale
            masked_lat = (self.vae.encode(masked_img).latent_dist.sample() - shift) * scale

        B, C, Hlat, Wlat = lat.shape
        T = getattr(self.scheduler.config, "num_train_timesteps", len(self.scheduler.timesteps))
        t_idx = torch.randint(0, T, (B,), device=self.device)
        sigmas = self.scheduler.sigmas[t_idx].view(B, 1, 1, 1)
        noise = torch.randn_like(lat)
        noisy_lat = (1 - sigmas) * lat + sigmas * noise

        mask_small = F.interpolate(mask, size=(Hlat, Wlat), mode="nearest")
        mask_bc = mask_small.expand(B, C, Hlat, Wlat)
        noisy_input_lat = lat * (1 - mask_bc) + noisy_lat * mask_bc

        packed_noisy = self.pipe._pack_latents(noisy_input_lat, B, C, Hlat, Wlat)
        packed_masked = self.pipe._pack_latents(masked_lat, B, C, Hlat, Wlat)
        packed_mask = self.pipe._pack_latents(
            _pack_mask(mask, lat, vae_scale), B, vae_scale * vae_scale, Hlat, Wlat
        )

        masked_image_latents = torch.cat((packed_masked, packed_mask), dim=2)
        transformer_input = torch.cat((packed_noisy, masked_image_latents), dim=2)

        prompt_embeds = torch.zeros(B, 1, 4096, device=self.device, dtype=transformer_input.dtype)
        pooled_prompt_embeds = torch.zeros(B, 768, device=self.device, dtype=transformer_input.dtype)
        text_ids = torch.zeros(1, 3, dtype=torch.long, device=self.device)
        latent_image_ids = self.pipe._prepare_latent_image_ids(
            B, Hlat // 2, Wlat // 2, self.device, torch.long
        )

        guidance = None
        if getattr(self.transformer.config, "guidance_embeds", False):
            guidance = torch.zeros(B, device=self.device, dtype=transformer_input.dtype)

        model_pred = self.transformer(
            hidden_states=transformer_input,
            timestep=self.scheduler.timesteps[t_idx].to(self.device) / 1000.0,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            return_dict=False,
        )[0]
        model_pred = FluxFillPipeline._unpack_latents(
            model_pred, height=Hlat * vae_scale, width=Wlat * vae_scale, vae_scale_factor=vae_scale
        )

        target = (noise - lat).to(model_pred.dtype)
        loss = F.mse_loss((model_pred * mask_bc).float(), (target * mask_bc).float())

        if train:
            self.acc.backward(loss)
            self.opt.step()
            self.opt.zero_grad()
        return loss

    @torch.no_grad()
    def validate(self) -> float:
        self.transformer.eval()
        losses = [self._step(batch, train=False).item() for batch in self.val_dl]
        self.transformer.train()
        return float(np.mean(losses)) if losses else 0.0

    def train(self, epochs: int) -> None:
        for ep in range(epochs):
            self.transformer.train()
            losses = []
            for batch in tqdm(self.dl, desc=f"epoch {ep + 1}/{epochs}"):
                losses.append(self._step(batch, train=True).item())

            val_loss = self.validate()
            print(f"epoch {ep + 1}: train={np.mean(losses):.6f} val={val_loss:.6f}")

            from peft.utils import get_peft_model_state_dict

            lora_state = get_peft_model_state_dict(self.transformer)
            self.pipe.save_lora_weights(str(self.save_dir / "lora"), transformer_lora_layers=lora_state)
