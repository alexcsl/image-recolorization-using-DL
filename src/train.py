"""
Unified Training & Test Evaluation Script for Image Colorization Experiments.

Usage:
  python -m src.train --config configs/baseline.yaml
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from src.dataset import ColorizationDataset
from src.metrics import ColorizationEvaluator
from src.models import build_model
from src.utils import count_parameters, lab_normalized_to_rgb, seed_everything


def build_optimizer(model: nn.Module, train_cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    opt_name = train_cfg["optimizer"].lower()
    lr = float(train_cfg["lr"])
    wd = float(train_cfg.get("weight_decay", 0.0))
    params = [p for p in model.parameters() if p.requires_grad]
    if opt_name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), weight_decay=wd)
    if opt_name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if opt_name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
    raise ValueError(f"Unsupported optimizer: {opt_name}")


def build_criterion(loss_name: str) -> nn.Module:
    name = loss_name.lower()
    if name == "mse":
        return nn.MSELoss()
    if name == "l1":
        return nn.L1Loss()
    raise ValueError(f"Unsupported loss: {loss_name}")


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    evaluator: ColorizationEvaluator | None = None,
    use_amp: bool = True,
) -> Dict[str, float]:
    model.eval()
    if evaluator is not None:
        evaluator.reset()

    total_loss = 0.0
    total_samples = 0

    for L, ab, _ in loader:
        L = L.to(device, non_blocking=True)
        ab = ab.to(device, non_blocking=True)
        bs = L.size(0)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
            pred_ab = model(L)
            loss = criterion(pred_ab, ab)

        total_loss += float(loss.item()) * bs
        total_samples += bs

        if evaluator is not None:
            pred_rgb = lab_normalized_to_rgb(L, pred_ab)
            gt_rgb = lab_normalized_to_rgb(L, ab)
            evaluator.update(pred_rgb, gt_rgb)

    out = {"loss": total_loss / max(total_samples, 1)}
    if evaluator is not None:
        out.update(evaluator.compute())
    return out


def upsert_metrics_summary(summary_path: Path, row: Dict[str, Any]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        df = df[df["experiment_name"] != row["experiment_name"]]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(summary_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a colorization experiment from YAML config.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--epochs", type=int, help="Override the configured epoch count for a pilot run.")
    args = parser.parse_args()
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be positive")

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs

    exp_name = cfg["experiment_name"]
    seed_everything(int(cfg.get("seed", 42)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"============================================================")
    print(f"Experiment : {exp_name}")
    print(f"Description: {cfg.get('description', '')}")
    print(f"Device     : {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"============================================================")

    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]

    splits_dir = Path(data_cfg["splits_dir"])
    img_dir = Path(data_cfg["processed_dir"])
    img_size = int(data_cfg.get("image_size", 256))
    num_workers = int(data_cfg.get("num_workers", 4))
    batch_size = int(train_cfg["batch_size"])

    train_ds = ColorizationDataset(splits_dir / "train.csv", img_dir, img_size, augment=True)
    val_ds = ColorizationDataset(splits_dir / "val.csv", img_dir, img_size, augment=False)
    test_ds = ColorizationDataset(splits_dir / "test.csv", img_dir, img_size, augment=False)

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=num_workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )

    model = build_model(model_cfg).to(device)
    total_params, trainable_params = count_parameters(model)
    print(f"[MODEL] Total params: {total_params:,} | Trainable: {trainable_params:,}")

    criterion = build_criterion(train_cfg["loss"])
    optimizer = build_optimizer(model, train_cfg)
    use_amp = bool(train_cfg.get("use_amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    evaluator = ColorizationEvaluator(device)
    epochs = int(train_cfg["epochs"])
    freeze_epochs = int(model_cfg.get("freeze_encoder_epochs", 0))

    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = ckpt_dir / f"{exp_name}_best.pt"

    history_dir = Path("results/history")
    history_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_epoch = 0
    history = []

    t_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        ep_start = time.perf_counter()

        if freeze_epochs > 0 and hasattr(model, "set_encoder_trainable"):
            if epoch == 1:
                model.set_encoder_trainable(False)
                optimizer = build_optimizer(model, train_cfg)
            elif epoch == freeze_epochs + 1:
                model.set_encoder_trainable(True)
                optimizer = build_optimizer(model, train_cfg)

        model.train()
        train_loss_sum = 0.0
        train_samples = 0

        for L, ab, _ in train_loader:
            L = L.to(device, non_blocking=True)
            ab = ab.to(device, non_blocking=True)
            bs = L.size(0)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                pred_ab = model(L)
                loss = criterion(pred_ab, ab)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += float(loss.item()) * bs
            train_samples += bs

        train_loss = train_loss_sum / max(train_samples, 1)

        # Compute full PSNR/SSIM/LPIPS on validation set every 5 epochs + epoch 1 + last epoch
        # and val loss every epoch (keeps training fast while giving smooth metric curves)
        compute_val_metrics = (epoch == 1) or (epoch % 5 == 0) or (epoch == epochs)
        val_res = evaluate_loader(
            model,
            val_loader,
            criterion,
            device,
            evaluator=evaluator if compute_val_metrics else None,
            use_amp=use_amp,
        )
        ep_time = time.perf_counter() - ep_start

        if val_res["loss"] < best_val_loss:
            best_val_loss = val_res["loss"]
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "config": cfg,
                    "val_loss": best_val_loss,
                },
                best_ckpt_path,
            )

        rec = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_res["loss"], 6),
            "val_psnr": round(val_res["psnr"], 4) if "psnr" in val_res else np.nan,
            "val_ssim": round(val_res["ssim"], 4) if "ssim" in val_res else np.nan,
            "val_lpips": round(val_res["lpips"], 4) if "lpips" in val_res else np.nan,
            "epoch_time_sec": round(ep_time, 2),
        }
        history.append(rec)
        pd.DataFrame(history).to_csv(history_dir / f"{exp_name}_history.csv", index=False)

        metric_str = (
            f" | val_psnr={val_res['psnr']:.2f} dB, val_ssim={val_res['ssim']:.4f}, val_lpips={val_res['lpips']:.4f}"
            if compute_val_metrics
            else ""
        )
        print(
            f"Epoch {epoch:02d}/{epochs:02d} ({ep_time:5.1f}s) | "
            f"train_loss={train_loss:.5f} | val_loss={val_res['loss']:.5f}{metric_str}"
        )

    total_train_time = time.perf_counter() - t_start

    # Load best checkpoint and run full test-set evaluation
    print(f"\n[EVAL] Loading best checkpoint (epoch {best_epoch}, val_loss={best_val_loss:.5f}) for test set...")
    ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_res = evaluate_loader(model, test_loader, criterion, device, evaluator=evaluator, use_amp=use_amp)

    print(
        f"[TEST] {exp_name} -> "
        f"Loss: {test_res['loss']:.5f} | "
        f"PSNR: {test_res['psnr']:.2f} dB | "
        f"SSIM: {test_res['ssim']:.4f} | "
        f"LPIPS: {test_res['lpips']:.4f} | "
        f"Train time: {total_train_time / 60:.2f} min"
    )

    summary_row = {
        "experiment_name": exp_name,
        "model": model_cfg["name"],
        "loss_fn": train_cfg["loss"],
        "optimizer": train_cfg["optimizer"],
        "lr": float(train_cfg["lr"]),
        "batch_size": batch_size,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "train_time_sec": round(total_train_time, 1),
        "train_time_min": round(total_train_time / 60.0, 2),
        "final_train_loss": round(history[-1]["train_loss"], 6),
        "best_val_loss": round(best_val_loss, 6),
        "test_loss": round(test_res["loss"], 6),
        "test_psnr": round(test_res["psnr"], 3),
        "test_ssim": round(test_res["ssim"], 4),
        "test_lpips": round(test_res["lpips"], 4),
    }
    upsert_metrics_summary(Path("results/metrics_summary.csv"), summary_row)

    # Also save per-run metadata JSON
    meta_path = history_dir / f"{exp_name}_summary.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(summary_row, f, indent=2)


if __name__ == "__main__":
    main()
