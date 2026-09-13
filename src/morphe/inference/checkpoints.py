"""Loading the published stage-1 checkpoint (UNet + CondEncoder + CoordEncoder).

``Hickey-Lab/MORPHE_CODEX_Outpainting`` is a directory saved by
``accelerator.save_state`` over the component order
``[vae, unet, coord_enc, cond_enc, ...]`` (see
:class:`morphe.training.latent_trainer.LatentTrainer`), so the four
``nn.Module``s serialise as ``model.safetensors`` (VAE), ``model_1.safetensors``
(UNet), ``model_2.safetensors`` (CoordEncoder) and ``model_3.safetensors``
(CondEncoder). Only those four files are needed for inference; the
``custom_checkpoint_*.pkl``, ``optimizer.bin``, ``random_states_*.pkl`` and
``scaler.pt`` in the published repository are optimizer/RNG state used only
to resume training.
"""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file

from morphe.models.cond_encoder import CondEncoder
from morphe.models.coord_encoder import CoordEncoder


def load_stage1_components(
    checkpoint_dir: str | Path,
    unet: torch.nn.Module,
    coord_encoder: CoordEncoder,
    cond_encoder: CondEncoder,
    device: torch.device | str = "cpu",
) -> None:
    """Load the fine-tuned UNet, CoordEncoder and CondEncoder weights in place."""
    checkpoint_dir = Path(checkpoint_dir)

    unet_state = load_file(checkpoint_dir / "model_1.safetensors", device=str(device))
    unet.load_state_dict(unet_state)

    coord_state = load_file(checkpoint_dir / "model_2.safetensors", device=str(device))
    coord_encoder.load_state_dict(coord_state)

    cond_state = load_file(checkpoint_dir / "model_3.safetensors", device=str(device))
    cond_encoder.load_state_dict(cond_state)
