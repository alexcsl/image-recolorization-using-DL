"""
Generate Comparison Plots & Visual Colorization Grids Across Experiments.

Usage:
  python -m src.evaluate --configs configs/baseline.yaml configs/e1_lr_1e3.yaml configs/e2_unet_resnet18.yaml
"""

import argparse
from pathlib import Path
from typing import List

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


def plot_curves(exp_names: List[str], labels: List[str], history_dir: Path, plot_dir: Path) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)

    # 1. Per-experiment Train vs Val Loss (clearly shows overfitting / underfitting / stability)
    fig, axes = plt.subplots(1, len(exp_names), figsize=(5.5 * len(exp_names), 4.2))
    if len(exp_names) == 1:
        axes = [axes]

    for ax, exp, lbl in zip(axes, exp_names, labels):
        df = pd.read_csv(history_dir / f"{exp}_history.csv")
        ax.plot(df["epoch"], df["train_loss"], label="Train Loss", linewidth=2, color="#2B6CB0")
        ax.plot(df["epoch"], df["val_loss"], label="Val Loss", linewidth=2, color="#E53E3E", linestyle="--")
        ax.set_title(lbl, fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.suptitle("Training vs. Validation Loss Curves", fontsize=13)
    plt.tight_layout()
    plt.savefig(plot_dir / "train_val_loss_curves.png", dpi=160)
    plt.close(fig)

    # 2. Metric progression across experiments
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    colors = ["#4A5568", "#DD6B20", "#2F855A", "#3182CE", "#805AD5"]

    for i, (exp, lbl) in enumerate(zip(exp_names, labels)):
        df = pd.read_csv(history_dir / f"{exp}_history.csv").dropna(subset=["val_psnr"])
        c = colors[i % len(colors)]
        for ax, metric in zip(axes, ("psnr", "ssim", "lpips")):
            ax.plot(df["epoch"], df[f"val_{metric}"], marker="o", label=f"{lbl} val", color=c)
            if f"train_{metric}" in df:
                ax.plot(df["epoch"], df[f"train_{metric}"], marker=".", linestyle="--",
                        label=f"{lbl} train sample", color=c)

    axes[0].set_title("PSNR (dB) ↑")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].set_title("SSIM ↑")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("SSIM")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    axes[2].set_title("LPIPS ↓")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("LPIPS")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=8)

    plt.suptitle("Training Sample and Validation Metrics", fontsize=13)
    plt.tight_layout()
    plt.savefig(plot_dir / "val_metrics_comparison.png", dpi=160)
    plt.close(fig)


@torch.no_grad()
def plot_visual_comparison(
    config_paths: List[Path],
    labels: List[str],
    out_path: Path,
    num_samples: int = 6,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(config_paths[0], "r", encoding="utf-8") as f:
        first_cfg = yaml.safe_load(f)

    test_ds = ColorizationDataset(
        Path(first_cfg["data"]["splits_dir"]) / "test.csv",
        Path(first_cfg["data"]["processed_dir"]),
        int(first_cfg["data"].get("image_size", 256)),
        augment=False,
    )

    # Pick evenly spaced indices from test set for diverse scenes
    indices = np.linspace(0, len(test_ds) - 1, num_samples, dtype=int)
    L_batch = torch.stack([test_ds[i][0] for i in indices]).to(device)
    ab_gt_batch = torch.stack([test_ds[i][1] for i in indices]).to(device)

    gt_rgb = lab_normalized_to_rgb(L_batch, ab_gt_batch)
    gray_rgb = ((L_batch.cpu().numpy()[:, 0] + 1.0) * 0.5).clip(0.0, 1.0)

    preds_per_model = []
    for cfg_path in config_paths:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        exp_name = cfg["experiment_name"]
        model = build_model(cfg["model"]).to(device)
        ckpt = torch.load(f"checkpoints/{exp_name}_best.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        pred_ab = model(L_batch)
        preds_per_model.append(lab_normalized_to_rgb(L_batch, pred_ab))

    n_cols = 2 + len(config_paths)
    col_titles = ["Grayscale (L)"] + labels + ["Ground Truth"]

    fig, axes = plt.subplots(num_samples, n_cols, figsize=(3.2 * n_cols, 3.0 * num_samples))
    for r in range(num_samples):
        axes[r, 0].imshow(gray_rgb[r], cmap="gray", vmin=0, vmax=1)
        for m_idx, pred_rgb in enumerate(preds_per_model):
            axes[r, 1 + m_idx].imshow(pred_rgb[r])
        axes[r, -1].imshow(gt_rgb[r])

        for c in range(n_cols):
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(col_titles[c], fontsize=11, fontweight="bold")

    plt.suptitle("Qualitative Test-Set Colorization Comparison", fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[DONE] Saved visual comparison grid -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate comparison plots & visual grids.")
    parser.add_argument("--configs", nargs="+", type=Path, required=True)
    parser.add_argument("--labels", nargs="+", type=str, default=None)
    parser.add_argument("--num-samples", type=int, default=6)
    args = parser.parse_args()

    exp_names = []
    for p in args.configs:
        with open(p, "r", encoding="utf-8") as f:
            exp_names.append(yaml.safe_load(f)["experiment_name"])

    labels = args.labels if args.labels and len(args.labels) == len(exp_names) else exp_names

    plot_curves(exp_names, labels, Path("results/history"), Path("results/plots"))
    plot_visual_comparison(
        args.configs,
        labels,
        Path("results/samples/comparison_grid.png"),
        num_samples=args.num_samples,
    )


if __name__ == "__main__":
    main()
