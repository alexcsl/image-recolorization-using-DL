"""
Compare finished experiments with plots and a side-by-side colorization grid.

Reads `results/history/<experiment>_history.csv` and `checkpoints/<experiment>_best.pt` for every
config and writes:
  - results/plots/train_val_loss_curves.png    train vs. validation loss per experiment
  - results/plots/val_metrics_comparison.png   PSNR / SSIM / LPIPS over epochs
  - results/samples/comparison_grid.png        the same test images colorized by every model

Usage:
    python -m src.evaluate --configs configs/baseline.yaml configs/e2_unet_resnet18.yaml configs/m3_palette.yaml
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from src.dataset import ColorizationDataset
from src.models import build_model
from src.utils import lab_normalized_to_rgb

HISTORY_DIR = Path("results/history")
PLOT_DIR = Path("results/plots")
GRID_PATH = Path("results/samples/comparison_grid.png")
CHECKPOINT_DIR = Path("checkpoints")

COLORS = ["#4A5568", "#DD6B20", "#2F855A", "#3182CE", "#805AD5"]
METRICS = [("psnr", "PSNR (dB) ↑", "PSNR (dB)"), ("ssim", "SSIM ↑", "SSIM"), ("lpips", "LPIPS ↓", "LPIPS")]


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Curves
# ---------------------------------------------------------------------------

def plot_loss_curves(exp_names: List[str], labels: List[str]) -> None:
    """One panel per experiment: a growing gap between train and val loss means overfitting."""
    fig, axes = plt.subplots(1, len(exp_names), figsize=(5.5 * len(exp_names), 4.2))
    axes = np.atleast_1d(axes)

    for ax, exp, label in zip(axes, exp_names, labels):
        history = pd.read_csv(HISTORY_DIR / f"{exp}_history.csv")
        ax.plot(history["epoch"], history["train_loss"], label="Train Loss", linewidth=2, color="#2B6CB0")
        ax.plot(history["epoch"], history["val_loss"], label="Val Loss", linewidth=2, color="#E53E3E", linestyle="--")
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.suptitle("Training vs. Validation Loss Curves", fontsize=13)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "train_val_loss_curves.png", dpi=160)
    plt.close(fig)


def plot_metric_curves(exp_names: List[str], labels: List[str]) -> None:
    """PSNR, SSIM, and LPIPS over epochs for all experiments (validation solid, train sample dashed)."""
    fig, axes = plt.subplots(1, len(METRICS), figsize=(15, 4.3))

    for i, (exp, label) in enumerate(zip(exp_names, labels)):
        history = pd.read_csv(HISTORY_DIR / f"{exp}_history.csv").dropna(subset=["val_psnr"])
        color = COLORS[i % len(COLORS)]
        for ax, (metric, _, _) in zip(axes, METRICS):
            ax.plot(history["epoch"], history[f"val_{metric}"], marker="o", label=f"{label} val", color=color)
            if f"train_{metric}" in history:
                ax.plot(history["epoch"], history[f"train_{metric}"], marker=".", linestyle="--",
                        label=f"{label} train sample", color=color)

    for ax, (_, title, ylabel) in zip(axes, METRICS):
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.suptitle("Training Sample and Validation Metrics", fontsize=13)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "val_metrics_comparison.png", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Qualitative grid
# ---------------------------------------------------------------------------

def load_trained_model(cfg: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    model = build_model(cfg["model"]).to(device)
    checkpoint = torch.load(CHECKPOINT_DIR / f"{cfg['experiment_name']}_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.eval()


@torch.no_grad()
def colorize(model: torch.nn.Module, L: torch.Tensor, seed: int = 42) -> torch.Tensor:
    """Predict ab for a batch of L. Diffusion models use a fixed seed so the result is repeatable."""
    if getattr(model, "is_diffusion", False):
        return model(L, generator=torch.Generator(device=L.device).manual_seed(seed))
    return model(L)


@torch.no_grad()
def plot_visual_comparison(config_paths: List[Path], labels: List[str], out_path: Path, num_samples: int = 6) -> None:
    """Rows are test images; columns are grayscale input, every model's output, and the ground truth."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs = [load_yaml(p) for p in config_paths]

    data_cfg = configs[0]["data"]
    test_ds = ColorizationDataset(
        Path(data_cfg["splits_dir"]) / "test.csv",
        Path(data_cfg["processed_dir"]),
        int(data_cfg.get("image_size", 256)),
        augment=False,
    )

    # Evenly spaced test images give a variety of scenes.
    indices = np.linspace(0, len(test_ds) - 1, num_samples, dtype=int)
    L = torch.stack([test_ds[i][0] for i in indices]).to(device)
    ab_true = torch.stack([test_ds[i][1] for i in indices]).to(device)

    gray = ((L.cpu().numpy()[:, 0] + 1.0) * 0.5).clip(0.0, 1.0)
    ground_truth = lab_normalized_to_rgb(L, ab_true)
    predictions = [
        lab_normalized_to_rgb(L, colorize(load_trained_model(cfg, device), L, int(cfg.get("seed", 42))))
        for cfg in configs
    ]

    columns = [gray] + predictions + [ground_truth]
    titles = ["Grayscale (L)"] + labels + ["Ground Truth"]

    fig, axes = plt.subplots(num_samples, len(columns), figsize=(3.2 * len(columns), 3.0 * num_samples))
    for row in range(num_samples):
        for col, images in enumerate(columns):
            ax = axes[row, col]
            if col == 0:
                ax.imshow(images[row], cmap="gray", vmin=0, vmax=1)
            else:
                ax.imshow(images[row])
            ax.axis("off")
            if row == 0:
                ax.set_title(titles[col], fontsize=11, fontweight="bold")

    plt.suptitle("Qualitative Test-Set Colorization Comparison", fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[DONE] Saved visual comparison grid -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate comparison plots and a visual grid.")
    parser.add_argument("--configs", nargs="+", type=Path, required=True)
    parser.add_argument("--labels", nargs="+", type=str, default=None, help="Display names, one per config.")
    parser.add_argument("--num-samples", type=int, default=6)
    args = parser.parse_args()

    exp_names = [load_yaml(p)["experiment_name"] for p in args.configs]
    labels = args.labels if args.labels and len(args.labels) == len(exp_names) else exp_names

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plot_loss_curves(exp_names, labels)
    plot_metric_curves(exp_names, labels)
    plot_visual_comparison(args.configs, labels, GRID_PATH, num_samples=args.num_samples)


if __name__ == "__main__":
    main()
