from typing import Any, Dict

import torch.nn as nn

from src.models.baseline_ae import SimpleCNNAutoencoder
from src.models.unet_resnet18 import UNetResNet18


def build_model(model_cfg: Dict[str, Any]) -> nn.Module:
    name = model_cfg["name"].lower()
    if name == "baseline_ae":
        return SimpleCNNAutoencoder(base_channels=int(model_cfg.get("base_channels", 64)))
    if name == "unet_resnet18":
        return UNetResNet18(pretrained=bool(model_cfg.get("pretrained", True)))
    raise ValueError(f"Unknown model name: {name}")
