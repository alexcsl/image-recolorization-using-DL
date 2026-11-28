"""
Model factory: turns the `model:` section of an experiment config into a model instance.

Every model takes the L channel (B, 1, H, W) and returns ab (B, 2, H, W), both in [-1, 1].

| name            | model                                   | proposal role |
| --------------- | --------------------------------------- | ------------- |
| `baseline_ae`   | SimpleCNNAutoencoder                    | Baseline      |
| `unet_resnet18` | U-Net with pretrained ResNet-18 encoder | Method 1      |
| `palette`       | Conditional diffusion with DDIM         | Method 3      |
"""

from typing import Any, Dict

import torch.nn as nn

from src.models.baseline_ae import SimpleCNNAutoencoder
from src.models.palette import PaletteColorizer
from src.models.unet_resnet18 import UNetResNet18


def build_palette(cfg: Dict[str, Any]) -> PaletteColorizer:
    return PaletteColorizer(
        # Diffusion process
        timesteps=int(cfg.get("timesteps", 1000)),
        beta_start=float(cfg.get("beta_start", 1e-4)),
        beta_end=float(cfg.get("beta_end", 0.02)),
        sampling_steps=int(cfg.get("sampling_steps", 50)),
        ddim_eta=float(cfg.get("ddim_eta", 0.0)),
        # Noise-prediction U-Net
        base_channels=int(cfg.get("base_channels", 64)),
        channel_mults=tuple(cfg.get("channel_mults", (1, 2, 4, 4))),
        num_res_blocks=int(cfg.get("num_res_blocks", 2)),
        attention_resolutions=tuple(cfg.get("attention_resolutions", (32,))),
        num_heads=int(cfg.get("num_heads", 4)),
        dropout=float(cfg.get("dropout", 0.1)),
        image_size=int(cfg.get("image_size", 256)),
    )


def build_model(model_cfg: Dict[str, Any]) -> nn.Module:
    name = model_cfg["name"].lower()
    if name == "baseline_ae":
        return SimpleCNNAutoencoder(base_channels=int(model_cfg.get("base_channels", 64)))
    if name == "unet_resnet18":
        return UNetResNet18(pretrained=bool(model_cfg.get("pretrained", True)))
    if name == "palette":
        return build_palette(model_cfg)
    raise ValueError(f"Unknown model name: {name}")
