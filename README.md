# Black-and-White Image Colorization with Deep Learning

Deep Learning Course Final Project
Repository: https://github.com/alexcsl/image-recolorization-using-DL

## Project Overview
Automatic image colorization predicts plausible, natural colors for grayscale photographs by learning semantic object context (sky, skin, foliage, clothing, roads) in **CIELAB** color space:
- **Input**: Lightness channel `L` (`1 × 256 × 256`, normalized to `[-1, 1]`)
- **Target**: Chrominance channels `a, b` (`2 × 256 × 256`, normalized to `[-1, 1]`)
- **Dataset**: **12,000** color images sampled deterministically (`seed=42`) from **MS COCO 2017 (`train2017`)** after filtering out natively grayscale/monochrome images, split **80% Train (9,600) / 10% Val (1,200) / 10% Test (1,200)**.
- **Metrics**: **PSNR** (↑), **SSIM** (↑), and **LPIPS** (↓).

---

## Repository Structure
```text
.
├── configs/                        # YAML experiment configs (single-variable controlled)
│   ├── baseline.yaml               # Simple CNN Autoencoder (MSE, Adam, lr=2e-4, bs=16, 30ep)
│   ├── e1_lr_1e3.yaml              # E1: Learning rate ablation (lr=1e-3)
│   ├── e2_unet_resnet18.yaml       # E2: U-Net + Pretrained ResNet-18 Encoder (L1 loss)
│   ├── e3_unet_adamw.yaml          # E3 (Ready): U-Net ResNet-18 with AdamW
│   └── e4_unet_bs32.yaml           # E4 (Ready): U-Net ResNet-18 with batch_size=32
├── data/
│   └── splits/                     # Committed reproducible manifests (train/val/test.csv)
├── notebooks/
│   ├── 01_eda_and_data_prep.ipynb          # EDA, preprocessing & split verification
│   └── 02_baseline_and_experiments.ipynb   # Model training, curves & qualitative eval
├── results/
│   ├── history/                    # Per-epoch CSV logs & per-run JSON summaries
│   ├── plots/                      # EDA charts, train/val loss & metric curves
│   ├── samples/                    # Side-by-side test-set colorization grids
│   ├── metrics_summary.csv         # Consolidated results table
├── scripts/
│   └── prepare_dataset.py          # Reproducible COCO 2017 downloader & splitter
├── src/
│   ├── dataset.py                  # CIELAB PyTorch Dataset + horizontal flip augmentation
│   ├── metrics.py                  # PSNR, SSIM, LPIPS evaluator
│   ├── models/
│   │   ├── baseline_ae.py          # Simple CNN Autoencoder (no skip, no pretrained)
│   │   └── unet_resnet18.py        # U-Net with ImageNet-pretrained ResNet-18 encoder
│   ├── train.py                    # Unified training + best-checkpoint test evaluation
│   ├── evaluate.py                 # Curve plots + qualitative comparison grid generator
│   └── utils.py                    # Seeding, CIELAB<->RGB transforms, param counter
├── requirements.txt
└── README.md
```

---

## Quickstart for Teammates

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/alexcsl/image-recolorization-using-DL.git
cd image-recolorization-using-DL

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# Install PyTorch (CUDA 12.4 for NVIDIA GPUs; omit --index-url on CPU/Mac)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### 2. Prepare the 12,000-Image Dataset Split
Raw images are gitignored. Once the split manifests are available, download the exact images listed in `data/splits/{train,val,test}.csv`:
```bash
python scripts/prepare_dataset.py --from-splits
```

To create the manifests on a fresh checkout before they are published:

```bash
python scripts/prepare_dataset.py --target-count 12000 --seed 42
```

Use the published manifests for comparable runs. An unavailable image during initial collection can change subset membership.

### 3. Run an Experiment
```bash
python -m src.train --config configs/baseline.yaml
python -m src.train --config configs/e1_lr_1e3.yaml
python -m src.train --config configs/e2_unet_resnet18.yaml
```

### 4. Regenerate Comparison Plots & Visual Grids
```bash
python -m src.evaluate --configs configs/baseline.yaml configs/e1_lr_1e3.yaml configs/e2_unet_resnet18.yaml
```

---

## Experiment Comparisons

| Run | Reference | Changed variable | Purpose |
| --- | --- | --- | --- |
| Baseline | None | CNN autoencoder, MSE, Adam, learning rate 0.0002, batch size 16 | Establish a reference for chrominance regression |
| E1 | Baseline | Learning rate 0.001 | Measure sensitivity to larger updates |
| E2 | Baseline | Pretrained ResNet-18 U-Net and L1 loss | Benchmark the proposed method as a package |
| E3 | E2 | AdamW optimizer with weight decay held at zero | Compare optimizers with other settings fixed |
| E4 | E2 | Batch size 32 | Measure the effect of averaging gradients over more images |

All runs use the same train, validation, and test manifests. The three single-variable experiments are E1, E3, and E4. E2 is a combined benchmark and cannot isolate the effect of its architecture or loss.

## Initial Pilot

Baseline and E1 completed five epochs on a shared 1,200-image COCO split (960 train, 120 validation, 120 test). The [pilot metrics](results/pilot/results/metrics_summary.csv), [loss curves](results/pilot/results/plots/train_val_loss_curves.png), and [metric curves](results/pilot/results/plots/val_metrics_comparison.png) document this smaller run. The main experiment configurations use 12,000 images and 30 epochs.

To download the exact pilot images from the repository root:

```bash
python scripts/prepare_dataset.py --from-splits --splits-dir results/pilot/splits
```

Run the pilot from `results/pilot` with the repository root on `PYTHONPATH`:

```powershell
cd results/pilot
$env:PYTHONPATH = (Resolve-Path ../..).Path
python -m src.train --config baseline.yaml
python -m src.train --config e1_lr_1e3.yaml
python -m src.evaluate --configs baseline.yaml e1_lr_1e3.yaml
```

## Team Git Workflow
1. Always create a feature branch for new experiments or models:
   ```bash
   git checkout -b exp/e3-adamw
   ```
2. Add a new YAML file in `configs/` changing **only one variable** per scenario.
3. Run `python -m src.train --config configs/<your_config>.yaml`.
4. Commit `configs/`, `results/history/`, `results/metrics_summary.csv`, and any new plots (never commit `data/processed_256/` or `checkpoints/`).
5. Open a Pull Request into `main`.
