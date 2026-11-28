"""
Train and evaluate one colorization experiment described by a YAML config.

Pipeline
--------
1. Read the YAML config and apply command-line overrides.
2. Build the train, validation, and test data loaders.
3. Build the model, loss, optimizer, and (optionally) an EMA copy of the model.
4. Train for the configured number of epochs. After every epoch:
     - compute the validation loss, plus PSNR/SSIM/LPIPS on epochs 1, 5, 10, ... and the last one,
     - keep the checkpoint with the lowest validation loss (`<name>_best.pt`),
     - save a full training state (`<name>_last.pt`) so an interrupted run can be resumed.
5. Reload the best checkpoint and evaluate it on the full test set (PSNR, SSIM, LPIPS, and FID).
6. Write the results to `results/metrics_summary.csv` and `results/history/`.

Usage
-----
    python -m src.train --config configs/baseline.yaml
    python -m src.train --config configs/m3_palette.yaml --resume
    python -m src.train --config configs/m3_palette.yaml --eval-only --sampling-steps 250
"""

import argparse
import copy
import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Subset

from src.dataset import ColorizationDataset
from src.metrics import ColorizationEvaluator
from src.models import build_model
from src.utils import count_parameters, lab_normalized_to_rgb, seed_everything

CHECKPOINT_DIR = Path("checkpoints")
HISTORY_DIR = Path("results/history")
SUMMARY_CSV = Path("results/metrics_summary.csv")

# Number of training images used to measure train-set PSNR/SSIM/LPIPS (for over/underfitting curves).
TRAIN_METRIC_SUBSET = 256


# ---------------------------------------------------------------------------
# Command line and configuration
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a colorization experiment from a YAML config.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the experiment YAML file.")
    parser.add_argument("--epochs", type=int, help="Override the configured epoch count (e.g. for a pilot run).")
    parser.add_argument("--sampling-steps", type=int, help="Override the DDIM sampling steps of a diffusion model (Scenario 6).")
    parser.add_argument("--eval-only", action="store_true", help="Skip training and evaluate the best checkpoint on the test set.")
    parser.add_argument("--resume", action="store_true", help="Continue training from checkpoints/<experiment>_last.pt if it exists.")
    args = parser.parse_args()

    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.sampling_steps is not None and args.sampling_steps < 1:
        parser.error("--sampling-steps must be positive")
    return args


def load_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Read the YAML config and apply the command-line overrides."""
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.sampling_steps is not None:
        cfg["model"]["sampling_steps"] = args.sampling_steps
    return cfg


def results_name(cfg: Dict[str, Any], args: argparse.Namespace) -> str:
    """
    Name used for the results files.

    An evaluation-only run with a different number of sampling steps gets a `_ddim<N>` suffix,
    so it adds a new row instead of overwriting the row of the trained run.
    """
    name = cfg["experiment_name"]
    if args.eval_only and args.sampling_steps is not None:
        name = f"{name}_ddim{args.sampling_steps}"
    return name


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class DataLoaders:
    train: DataLoader          # shuffled, with horizontal-flip augmentation
    val: DataLoader            # full validation split
    train_sample: DataLoader   # fixed subset of training images, no augmentation (for train metrics)
    test: DataLoader           # full test split


def build_dataloaders(cfg: Dict[str, Any], device: torch.device) -> DataLoaders:
    data_cfg = cfg["data"]
    seed = int(cfg.get("seed", 42))
    splits_dir = Path(data_cfg["splits_dir"])
    image_dir = Path(data_cfg["processed_dir"])
    image_size = int(data_cfg.get("image_size", 256))
    batch_size = int(cfg["training"]["batch_size"])
    num_workers = int(data_cfg.get("num_workers", 4))
    if os.name == "nt":
        num_workers = min(num_workers, 2)  # Windows worker processes are slow to start
    pin_memory = device.type == "cuda"

    def dataset(split: str, augment: bool = False) -> ColorizationDataset:
        return ColorizationDataset(splits_dir / f"{split}.csv", image_dir, image_size, augment=augment)

    train_loader = DataLoader(
        dataset("train", augment=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        drop_last=True,
    )

    train_eval_ds = dataset("train")
    subset_rng = np.random.default_rng(seed)
    subset_indices = subset_rng.choice(
        len(train_eval_ds), min(TRAIN_METRIC_SUBSET, len(train_eval_ds)), replace=False
    )
    train_sample_loader = DataLoader(
        Subset(train_eval_ds, subset_indices.tolist()),
        batch_size=batch_size,
        pin_memory=pin_memory,
        generator=torch.Generator().manual_seed(seed),
    )

    return DataLoaders(
        train=train_loader,
        val=DataLoader(dataset("val"), batch_size=batch_size, shuffle=False, pin_memory=pin_memory),
        train_sample=train_sample_loader,
        test=DataLoader(dataset("test"), batch_size=batch_size, shuffle=False, pin_memory=pin_memory),
    )


# ---------------------------------------------------------------------------
# Model, loss, optimizer, EMA
# ---------------------------------------------------------------------------

def build_optimizer(model: nn.Module, train_cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    name = train_cfg["optimizer"].lower()
    lr = float(train_cfg["lr"])
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    params = [p for p in model.parameters() if p.requires_grad]
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


def build_criterion(loss_name: str) -> nn.Module:
    """Regression loss on the predicted ab channels. Diffusion models compute their own loss."""
    name = loss_name.lower()
    if name in ("mse", "l2"):
        return nn.MSELoss()
    if name == "l1":
        return nn.L1Loss()
    raise ValueError(f"Unsupported loss: {loss_name}")


def is_diffusion(model: nn.Module) -> bool:
    return bool(getattr(model, "is_diffusion", False))


def create_ema_model(model: nn.Module) -> nn.Module:
    """A frozen copy of the model whose weights track an exponential moving average of the trained weights."""
    ema_model = copy.deepcopy(model).eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)
    return ema_model


@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float, step: int) -> None:
    """
    ema = decay * ema + (1 - decay) * weights.

    The decay ramps up from 0.1 during the first steps, so the EMA model is useful early in training
    instead of staying close to the random initial weights.
    """
    decay = min(decay, (1.0 + step) / (10.0 + step))
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.lerp_(p.detach(), 1.0 - decay)
    for ema_b, b in zip(ema_model.buffers(), model.buffers()):
        ema_b.copy_(b)


@dataclass
class Experiment:
    """Everything one run needs, built once in `build_experiment`."""

    cfg: Dict[str, Any]
    name: str                          # results name (may carry a _ddim<N> suffix)
    checkpoint_name: str               # checkpoints always use the config's experiment_name
    device: torch.device
    loaders: DataLoaders
    model: nn.Module
    ema_model: Optional[nn.Module]
    criterion: nn.Module
    optimizer: torch.optim.Optimizer
    scaler: torch.amp.GradScaler
    evaluator: ColorizationEvaluator
    use_amp: bool
    seed: int

    @property
    def eval_model(self) -> nn.Module:
        """The weights used for validation, checkpoints, and testing."""
        return self.ema_model if self.ema_model is not None else self.model

    @property
    def best_checkpoint(self) -> Path:
        return CHECKPOINT_DIR / f"{self.checkpoint_name}_best.pt"

    @property
    def last_checkpoint(self) -> Path:
        return CHECKPOINT_DIR / f"{self.checkpoint_name}_last.pt"

    @property
    def freeze_epochs(self) -> int:
        return int(self.cfg["model"].get("freeze_encoder_epochs", 0))


def build_experiment(cfg: Dict[str, Any], name: str, device: torch.device) -> Experiment:
    # Keep this order: model and LPIPS initialization consume the global random state.
    loaders = build_dataloaders(cfg, device)

    model = build_model(cfg["model"]).to(device)
    total, trainable = count_parameters(model)
    print(f"[MODEL] Total params: {total:,} | Trainable: {trainable:,}")
    if is_diffusion(model):
        print(f"[MODEL] Diffusion: DDIM sampling with {model.sampling_steps} steps")

    train_cfg = cfg["training"]
    criterion = build_criterion(train_cfg["loss"])
    optimizer = build_optimizer(model, train_cfg)
    use_amp = bool(train_cfg.get("use_amp", True)) and device.type == "cuda"
    evaluator = ColorizationEvaluator(device)
    ema_model = create_ema_model(model) if float(train_cfg.get("ema_decay", 0.0)) > 0 else None

    return Experiment(
        cfg=cfg,
        name=name,
        checkpoint_name=cfg["experiment_name"],
        device=device,
        loaders=loaders,
        model=model,
        ema_model=ema_model,
        criterion=criterion,
        optimizer=optimizer,
        scaler=torch.amp.GradScaler("cuda", enabled=use_amp),
        evaluator=evaluator,
        use_amp=use_amp,
        seed=int(cfg.get("seed", 42)),
    )


def set_encoder_frozen(exp: Experiment, frozen: bool) -> None:
    """Freeze or unfreeze a pretrained encoder and rebuild the optimizer for the new trainable set."""
    exp.model.set_encoder_trainable(not frozen)
    exp.optimizer = build_optimizer(exp.model, exp.cfg["training"])


# ---------------------------------------------------------------------------
# One pass over the data
# ---------------------------------------------------------------------------

def train_one_epoch(exp: Experiment, global_step: int) -> Tuple[float, int]:
    """Run one epoch of training. Returns (mean training loss, updated global step)."""
    ema_decay = float(exp.cfg["training"].get("ema_decay", 0.0))
    exp.model.train()
    loss_sum, n_images = 0.0, 0

    for L, ab, _ in exp.loaders.train:
        L = L.to(exp.device, non_blocking=True)
        ab = ab.to(exp.device, non_blocking=True)

        exp.optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=exp.device.type, enabled=exp.use_amp):
            if is_diffusion(exp.model):
                loss = exp.model.diffusion_loss(L, ab)
            else:
                loss = exp.criterion(exp.model(L), ab)

        exp.scaler.scale(loss).backward()
        exp.scaler.step(exp.optimizer)
        exp.scaler.update()
        if exp.ema_model is not None:
            update_ema(exp.ema_model, exp.model, ema_decay, global_step)
        global_step += 1

        loss_sum += float(loss.item()) * L.size(0)
        n_images += L.size(0)

    return loss_sum / max(n_images, 1), global_step


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    evaluator: Optional[ColorizationEvaluator] = None,
    use_amp: bool = True,
    seed: int = 42,
    max_metric_samples: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute the mean loss over `loader`, and PSNR/SSIM/LPIPS (plus FID if the evaluator has it
    enabled) when an `evaluator` is given.

    For diffusion models the loss is the denoising loss, and colors are generated with DDIM sampling.
    A fixed `seed` makes both repeatable, so results from different epochs are comparable.
    `max_metric_samples` limits how many images are colorized for the metrics (sampling is slow).
    """
    model.eval()
    if evaluator is not None:
        evaluator.reset()

    diffusion = is_diffusion(model)
    generator = torch.Generator(device=device).manual_seed(seed) if diffusion else None
    loss_sum, n_images, n_metric_images = 0.0, 0, 0

    for L, ab, _ in loader:
        L = L.to(device, non_blocking=True)
        ab = ab.to(device, non_blocking=True)
        want_metrics = evaluator is not None and (
            max_metric_samples is None or n_metric_images < max_metric_samples
        )

        with torch.amp.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
            if diffusion:
                loss = model.diffusion_loss(L, ab, generator=generator)
                pred_ab = model(L, generator=generator) if want_metrics else None
            else:
                pred_ab = model(L)
                loss = criterion(pred_ab, ab)

        loss_sum += float(loss.item()) * L.size(0)
        n_images += L.size(0)

        if want_metrics:
            n_metric_images += L.size(0)
            evaluator.update(lab_normalized_to_rgb(L, pred_ab), lab_normalized_to_rgb(L, ab))

    results = {"loss": loss_sum / max(n_images, 1)}
    if evaluator is not None:
        results.update(evaluator.compute())
    return results


def evaluate(
    exp: Experiment,
    loader: DataLoader,
    with_metrics: bool,
    metric_limit: Optional[int],
    evaluator: Optional[ColorizationEvaluator] = None,
) -> Dict[str, float]:
    """Evaluate the EMA/eval model on `loader`; `evaluator` defaults to the experiment's (no FID)."""
    return evaluate_loader(
        exp.eval_model,
        loader,
        exp.criterion,
        exp.device,
        evaluator=(evaluator or exp.evaluator) if with_metrics else None,
        use_amp=exp.use_amp,
        seed=exp.seed,
        max_metric_samples=metric_limit,
    )


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

@dataclass
class TrainingState:
    """Progress that must survive an interruption."""

    epoch: int = 0
    global_step: int = 0
    best_val_loss: float = float("inf")
    best_epoch: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)
    train_time_sec: float = 0.0


def save_best_checkpoint(exp: Experiment, epoch: int, val_loss: float) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": exp.eval_model.state_dict(),
            "config": exp.cfg,
            "val_loss": val_loss,
        },
        exp.best_checkpoint,
    )


def save_last_checkpoint(exp: Experiment, state: TrainingState) -> None:
    """Save everything needed to continue training exactly where it stopped, including random states."""
    payload = {
        "epoch": state.epoch,
        "model_state_dict": exp.model.state_dict(),
        "ema_state_dict": exp.ema_model.state_dict() if exp.ema_model is not None else None,
        "optimizer_state_dict": exp.optimizer.state_dict(),
        "scaler_state_dict": exp.scaler.state_dict(),
        "global_step": state.global_step,
        "best_val_loss": state.best_val_loss,
        "best_epoch": state.best_epoch,
        "history": state.history,
        "train_time_sec": state.train_time_sec,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if exp.device.type == "cuda" else None,
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
        "config": exp.cfg,
    }
    tmp_path = exp.last_checkpoint.with_suffix(".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(exp.last_checkpoint)  # atomic: a crash never leaves a half-written checkpoint


def load_last_checkpoint(exp: Experiment) -> TrainingState:
    saved = torch.load(exp.last_checkpoint, map_location=exp.device, weights_only=False)
    exp.model.load_state_dict(saved["model_state_dict"])
    if exp.ema_model is not None:
        exp.ema_model.load_state_dict(saved["ema_state_dict"])

    next_epoch = int(saved["epoch"]) + 1
    if exp.freeze_epochs > 0 and hasattr(exp.model, "set_encoder_trainable"):
        set_encoder_frozen(exp, frozen=next_epoch <= exp.freeze_epochs)
    exp.optimizer.load_state_dict(saved["optimizer_state_dict"])
    exp.scaler.load_state_dict(saved["scaler_state_dict"])

    torch.set_rng_state(saved["torch_rng_state"])
    if exp.device.type == "cuda" and saved.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state_all(saved["cuda_rng_state"])
    np.random.set_state(saved["numpy_rng_state"])
    random.setstate(saved["python_rng_state"])

    return TrainingState(
        epoch=int(saved["epoch"]),
        global_step=int(saved["global_step"]),
        best_val_loss=float(saved["best_val_loss"]),
        best_epoch=int(saved["best_epoch"]),
        history=list(saved["history"]),
        train_time_sec=float(saved["train_time_sec"]),
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def should_compute_metrics(epoch: int, total_epochs: int) -> bool:
    """PSNR/SSIM/LPIPS are slow, so they run on epoch 1, every 5th epoch, and the last epoch."""
    return epoch == 1 or epoch % 5 == 0 or epoch == total_epochs


def history_record(epoch: int, train_loss: float, val: Dict[str, float], train: Dict[str, float], seconds: float) -> Dict[str, Any]:
    def metric(results: Dict[str, float], key: str) -> float:
        return round(results[key], 4) if key in results else np.nan

    return {
        "epoch": epoch,
        "train_loss": round(train_loss, 6),
        "val_loss": round(val["loss"], 6),
        "val_psnr": metric(val, "psnr"),
        "val_ssim": metric(val, "ssim"),
        "val_lpips": metric(val, "lpips"),
        "train_psnr": metric(train, "psnr"),
        "train_ssim": metric(train, "ssim"),
        "train_lpips": metric(train, "lpips"),
        "epoch_time_sec": round(seconds, 2),
    }


def train(exp: Experiment, state: TrainingState) -> None:
    """Train from `state.epoch + 1` up to the configured number of epochs."""
    total_epochs = int(exp.cfg["training"]["epochs"])
    metric_limit = exp.cfg["data"].get("val_metric_samples")
    metric_limit = int(metric_limit) if metric_limit else None
    can_freeze = exp.freeze_epochs > 0 and hasattr(exp.model, "set_encoder_trainable")

    time_before_session = state.train_time_sec
    session_start = time.perf_counter()

    for epoch in range(state.epoch + 1, total_epochs + 1):
        epoch_start = time.perf_counter()
        if can_freeze and epoch == 1:
            set_encoder_frozen(exp, frozen=True)
        elif can_freeze and epoch == exp.freeze_epochs + 1:
            set_encoder_frozen(exp, frozen=False)

        train_loss, state.global_step = train_one_epoch(exp, state.global_step)

        with_metrics = should_compute_metrics(epoch, total_epochs)
        val = evaluate(exp, exp.loaders.val, with_metrics, metric_limit)
        train_metrics = evaluate(exp, exp.loaders.train_sample, True, metric_limit) if with_metrics else {}
        epoch_seconds = time.perf_counter() - epoch_start

        if val["loss"] < state.best_val_loss:
            state.best_val_loss = val["loss"]
            state.best_epoch = epoch
            save_best_checkpoint(exp, epoch, val["loss"])

        state.history.append(history_record(epoch, train_loss, val, train_metrics, epoch_seconds))
        pd.DataFrame(state.history).to_csv(HISTORY_DIR / f"{exp.name}_history.csv", index=False)

        metric_text = (
            f" | val_psnr={val['psnr']:.2f} dB, val_ssim={val['ssim']:.4f}, val_lpips={val['lpips']:.4f}"
            if with_metrics
            else ""
        )
        print(
            f"Epoch {epoch:02d}/{total_epochs:02d} ({epoch_seconds:5.1f}s) | "
            f"train_loss={train_loss:.5f} | val_loss={val['loss']:.5f}{metric_text}"
        )

        state.epoch = epoch
        state.train_time_sec = time_before_session + time.perf_counter() - session_start
        save_last_checkpoint(exp, state)

    state.train_time_sec = time_before_session + time.perf_counter() - session_start


# ---------------------------------------------------------------------------
# Test evaluation and results
# ---------------------------------------------------------------------------

def test(exp: Experiment) -> Tuple[Dict[str, float], float, int, float]:
    """Load the best checkpoint and evaluate it on the full test set, including FID.

    FID compares the distribution of all colorized test images with the real ones, so it is computed
    only here (on all 1,200 test images) and not on the small validation subsets during training.
    The evaluator is created here so the Inception network is loaded only when it is needed.

    Returns (test results, test time in seconds, best epoch, best validation loss).
    """
    best = torch.load(exp.best_checkpoint, map_location=exp.device, weights_only=False)
    print(f"\n[EVAL] Loading best checkpoint (epoch {best['epoch']}, val_loss={best['val_loss']:.5f}) for test set...")
    exp.eval_model.load_state_dict(best["model_state_dict"])

    test_evaluator = ColorizationEvaluator(exp.device, with_fid=True)
    start = time.perf_counter()
    results = evaluate(exp, exp.loaders.test, with_metrics=True, metric_limit=None, evaluator=test_evaluator)
    return results, time.perf_counter() - start, int(best["epoch"]), float(best["val_loss"])


def read_trained_run_summary(checkpoint_name: str) -> Dict[str, Any]:
    """Summary JSON written when the checkpoint was trained (used by --eval-only runs)."""
    path = HISTORY_DIR / f"{checkpoint_name}_summary.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_summary_row(
    exp: Experiment,
    test_results: Dict[str, float],
    test_seconds: float,
    best_epoch: int,
    best_val_loss: float,
    train_seconds: float,
    epochs: int,
    final_train_loss: float,
) -> Dict[str, Any]:
    train_cfg = exp.cfg["training"]
    total_params, trainable_params = count_parameters(exp.model)
    return {
        "experiment_name": exp.name,
        "model": exp.cfg["model"]["name"],
        "loss_fn": train_cfg["loss"],
        "optimizer": train_cfg["optimizer"],
        "lr": float(train_cfg["lr"]),
        "batch_size": int(train_cfg["batch_size"]),
        "epochs": epochs,
        "sampling_steps": int(exp.cfg["model"]["sampling_steps"]) if is_diffusion(exp.model) else np.nan,
        "best_epoch": best_epoch,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "train_time_sec": round(train_seconds, 1),
        "train_time_min": round(train_seconds / 60.0, 2),
        "test_time_sec": round(test_seconds, 1),
        "final_train_loss": final_train_loss,
        "best_val_loss": round(best_val_loss, 6),
        "test_loss": round(test_results["loss"], 6),
        "test_psnr": round(test_results["psnr"], 3),
        "test_ssim": round(test_results["ssim"], 4),
        "test_lpips": round(test_results["lpips"], 4),
        "test_fid": round(test_results["fid"], 3),
    }


def save_results(row: Dict[str, Any]) -> None:
    """Insert or replace this run's row in metrics_summary.csv and write its summary JSON."""
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame([row])
    if SUMMARY_CSV.exists():
        previous = pd.read_csv(SUMMARY_CSV)
        previous = previous[previous["experiment_name"] != row["experiment_name"]]
        table = pd.concat([previous, table], ignore_index=True)
    table.to_csv(SUMMARY_CSV, index=False)

    with open(HISTORY_DIR / f"{row['experiment_name']}_summary.json", "w", encoding="utf-8") as f:
        json.dump(row, f, indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def print_header(cfg: Dict[str, Any], name: str, device: torch.device) -> None:
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print("=" * 60)
    print(f"Experiment : {name}")
    print(f"Description: {cfg.get('description', '')}")
    print(f"Device     : {device} ({gpu})")
    print("=" * 60)


def main() -> None:
    args = parse_args()
    cfg = load_config(args)
    name = results_name(cfg, args)
    seed_everything(int(cfg.get("seed", 42)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_header(cfg, name, device)

    exp = build_experiment(cfg, name, device)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    state = TrainingState()
    if args.eval_only:
        if not exp.best_checkpoint.exists():
            raise FileNotFoundError(f"--eval-only needs a trained checkpoint at {exp.best_checkpoint}")
    else:
        if args.resume and exp.last_checkpoint.exists():
            state = load_last_checkpoint(exp)
            print(f"[RESUME] Continuing from epoch {state.epoch + 1} (best epoch so far: {state.best_epoch})")
        elif args.resume:
            print(f"[RESUME] No checkpoint at {exp.last_checkpoint}; starting from epoch 1")
        train(exp, state)

    test_results, test_seconds, best_epoch, best_val_loss = test(exp)

    # Training facts come from this run, or from the original training run when only evaluating.
    epochs = int(cfg["training"]["epochs"])
    train_seconds = state.train_time_sec
    final_train_loss = round(state.history[-1]["train_loss"], 6) if state.history else np.nan
    if args.eval_only:
        trained = read_trained_run_summary(exp.checkpoint_name)
        epochs = int(trained.get("epochs", epochs))
        train_seconds = float(trained.get("train_time_sec", 0.0))
        final_train_loss = trained.get("final_train_loss", np.nan)

    print(
        f"[TEST] {name} -> "
        f"Loss: {test_results['loss']:.5f} | "
        f"PSNR: {test_results['psnr']:.2f} dB | "
        f"SSIM: {test_results['ssim']:.4f} | "
        f"LPIPS: {test_results['lpips']:.4f} | "
        f"FID: {test_results['fid']:.2f} | "
        f"Train time: {train_seconds / 60:.2f} min | "
        f"Test inference: {test_seconds:.1f} s"
    )

    save_results(build_summary_row(
        exp, test_results, test_seconds, best_epoch, best_val_loss,
        train_seconds, epochs, final_train_loss,
    ))


if __name__ == "__main__":
    main()
