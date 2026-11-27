import random
from typing import Tuple

import cv2
import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Set random seeds across Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def rgb_to_lab_normalized(rgb_uint8: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert an RGB uint8 image (H, W, 3) in [0, 255] to normalized CIELAB channels:
      - L_norm: (H, W) float32 in [-1, 1]  via (L / 50.0) - 1.0
      - ab_norm: (H, W, 2) float32 in [-1, 1] via ab / 128.0
    """
    rgb_float = rgb_uint8.astype(np.float32) / 255.0
    lab = cv2.cvtColor(rgb_float, cv2.COLOR_RGB2LAB)
    L_norm = (lab[:, :, 0] / 50.0) - 1.0
    ab_norm = lab[:, :, 1:3] / 128.0
    return L_norm.astype(np.float32), ab_norm.astype(np.float32)


def lab_normalized_to_rgb(L_norm: torch.Tensor, ab_norm: torch.Tensor) -> np.ndarray:
    """
    Convert batches of normalized L (B, 1, H, W) and ab (B, 2, H, W) in [-1, 1]
    to RGB numpy arrays of shape (B, H, W, 3) float32 in [0, 1].
    """
    L_np = L_norm.detach().cpu().float().numpy()
    ab_np = ab_norm.detach().cpu().float().numpy()
    batch_size = L_np.shape[0]
    rgb_list = []
    for i in range(batch_size):
        L = np.clip((L_np[i, 0] + 1.0) * 50.0, 0.0, 100.0)
        a = np.clip(ab_np[i, 0] * 128.0, -127.0, 127.0)
        b = np.clip(ab_np[i, 1] * 128.0, -127.0, 127.0)
        lab = np.stack([L, a, b], axis=-1).astype(np.float32)
        rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        rgb = np.clip(rgb, 0.0, 1.0)
        rgb_list.append(rgb)
    return np.stack(rgb_list, axis=0)


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """Return (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
