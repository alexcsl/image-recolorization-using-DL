"""
Image-quality metrics, computed on the full RGB image (input L + predicted ab):

- PSNR  (higher is better): pixel-level error in decibels.
- SSIM  (higher is better): similarity of local structure and contrast, closer to human perception.
- LPIPS (lower is better):  distance between deep (AlexNet) features; tolerant of a different but
                            plausible color, e.g. a red car predicted as blue.
- FID   (lower is better):  Fréchet Inception Distance between the *set* of colorized images and the
                            set of real images. Every image is turned into a 2048-d Inception-v3 feature
                            vector; each set is summarized by its mean and covariance, and FID is the
                            distance between the two Gaussians:
                                FID = ||mu_real - mu_fake||^2 + Tr(S_real + S_fake - 2 (S_real S_fake)^(1/2))
                            Unlike the other metrics it does not compare image pairs, so it rewards
                            colors that look realistic overall, even if they differ from the original.

PSNR, SSIM, and LPIPS are averages over images. FID is one number for the whole set and is only
reliable with many images (it is computed on the full test set), so it is optional.
"""

from typing import Dict, List

import lpips
import numpy as np
import torch
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


class InceptionFeatures:
    """Pool-3 features (2048-d) of the Inception-v3 network used by the standard FID implementation."""

    def __init__(self, device: torch.device) -> None:
        block = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        # resize_input: images are resized to 299x299; normalize_input: [0, 1] is scaled to [-1, 1].
        self.model = InceptionV3([block], resize_input=True, normalize_input=True).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, images: torch.Tensor) -> np.ndarray:
        """images: (B, 3, H, W) float in [0, 1] -> (B, 2048) float64 features."""
        features = self.model(images.to(self.device, dtype=torch.float32))[0]
        return features.squeeze(-1).squeeze(-1).cpu().double().numpy()


def frechet_inception_distance(real_features: np.ndarray, fake_features: np.ndarray) -> float:
    mu_real, sigma_real = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu_fake, sigma_fake = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
    distance = float(calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake))
    return max(distance, 0.0)  # identical sets can give a tiny negative value from rounding


def to_tensor(images: np.ndarray, device: torch.device) -> torch.Tensor:
    """(B, H, W, 3) numpy in [0, 1] -> (B, 3, H, W) tensor in [0, 1]."""
    return torch.from_numpy(images).permute(0, 3, 1, 2).to(device, non_blocking=True)


class ColorizationEvaluator:
    """
    Accumulates PSNR, SSIM, LPIPS (and optionally FID) over batches of RGB images in [0, 1].

    Usage:
        evaluator.reset()
        for each batch: evaluator.update(pred_rgb, gt_rgb)
        results = evaluator.compute()   # {"psnr", "ssim", "lpips"} (+ "fid" when with_fid=True)
    """

    def __init__(self, device: torch.device, with_fid: bool = False) -> None:
        self.device = device
        self.lpips_fn = lpips.LPIPS(net="alex").to(device)
        self.lpips_fn.eval()
        for p in self.lpips_fn.parameters():
            p.requires_grad = False
        self.inception = InceptionFeatures(device) if with_fid else None
        self.reset()

    def reset(self) -> None:
        self.psnr_sum = 0.0
        self.ssim_sum = 0.0
        self.lpips_sum = 0.0
        self.count = 0
        self.real_features: List[np.ndarray] = []
        self.fake_features: List[np.ndarray] = []

    @torch.no_grad()
    def update(self, pred_rgb: np.ndarray, gt_rgb: np.ndarray) -> None:
        """pred_rgb, gt_rgb: numpy arrays of shape (B, H, W, 3), float32 in [0, 1]."""
        for pred, gt in zip(pred_rgb, gt_rgb):
            self.psnr_sum += float(peak_signal_noise_ratio(gt, pred, data_range=1.0))
            self.ssim_sum += float(structural_similarity(gt, pred, channel_axis=2, data_range=1.0))

        pred_t = to_tensor(pred_rgb, self.device)
        gt_t = to_tensor(gt_rgb, self.device)

        # LPIPS expects images in [-1, 1].
        self.lpips_sum += float(self.lpips_fn(pred_t * 2.0 - 1.0, gt_t * 2.0 - 1.0).sum().item())
        self.count += pred_rgb.shape[0]

        if self.inception is not None:
            self.fake_features.append(self.inception(pred_t))
            self.real_features.append(self.inception(gt_t))

    def compute(self) -> Dict[str, float]:
        if self.count == 0:
            return {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0}
        results = {
            "psnr": self.psnr_sum / self.count,
            "ssim": self.ssim_sum / self.count,
            "lpips": self.lpips_sum / self.count,
        }
        if self.inception is not None and self.count >= 2:
            results["fid"] = frechet_inception_distance(
                np.concatenate(self.real_features), np.concatenate(self.fake_features)
            )
        return results
