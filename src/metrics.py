from typing import Dict

import lpips
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


class ColorizationEvaluator:
    """Computes PSNR, SSIM, and LPIPS (AlexNet) over batches of RGB images in [0, 1]."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.lpips_fn = lpips.LPIPS(net="alex").to(device)
        self.lpips_fn.eval()
        for p in self.lpips_fn.parameters():
            p.requires_grad = False
        self.reset()

    def reset(self) -> None:
        self.psnr_sum = 0.0
        self.ssim_sum = 0.0
        self.lpips_sum = 0.0
        self.count = 0

    @torch.no_grad()
    def update(self, pred_rgb: np.ndarray, gt_rgb: np.ndarray) -> None:
        """
        pred_rgb, gt_rgb: numpy arrays of shape (B, H, W, 3) float32 in [0, 1].
        """
        batch_size = pred_rgb.shape[0]
        for i in range(batch_size):
            self.psnr_sum += float(
                peak_signal_noise_ratio(gt_rgb[i], pred_rgb[i], data_range=1.0)
            )
            self.ssim_sum += float(
                structural_similarity(
                    gt_rgb[i], pred_rgb[i], channel_axis=2, data_range=1.0
                )
            )

        # LPIPS expects (B, 3, H, W) float tensors in [-1, 1]
        pred_t = (
            torch.from_numpy(pred_rgb)
            .permute(0, 3, 1, 2)
            .to(self.device, non_blocking=True)
            * 2.0
            - 1.0
        )
        gt_t = (
            torch.from_numpy(gt_rgb)
            .permute(0, 3, 1, 2)
            .to(self.device, non_blocking=True)
            * 2.0
            - 1.0
        )
        lpips_vals = self.lpips_fn(pred_t, gt_t).view(-1)
        self.lpips_sum += float(lpips_vals.sum().item())
        self.count += batch_size

    def compute(self) -> Dict[str, float]:
        if self.count == 0:
            return {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0}
        return {
            "psnr": self.psnr_sum / self.count,
            "ssim": self.ssim_sum / self.count,
            "lpips": self.lpips_sum / self.count,
        }
