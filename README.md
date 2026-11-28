# Black-and-White Image Colorization with Deep Learning

Deep Learning Course Final Project
Repository: https://github.com/alexcsl/image-recolorization-using-DL

## Project Overview
Automatic image colorization predicts plausible, natural colors for grayscale photographs by learning semantic object context (sky, skin, foliage, clothing, roads) in **CIELAB** color space:
- **Input**: Lightness channel `L` (`1 × 256 × 256`, normalized to `[-1, 1]`)
- **Target**: Chrominance channels `a, b` (`2 × 256 × 256`, normalized to `[-1, 1]`)
- **Dataset**: **12,000** color images sampled deterministically (`seed=42`) from **MS COCO 2017 (`train2017`)** after filtering out natively grayscale/monochrome images, split **80% Train (9,600) / 10% Val (1,200) / 10% Test (1,200)**.
- **Metrics**: **PSNR** (↑), **SSIM** (↑), **LPIPS** (↓), and **FID** (↓).
  - PSNR, SSIM, and LPIPS compare each colorized image with its original and are averaged over images.
  - FID (Fréchet Inception Distance) compares the distribution of all colorized test images with the distribution of the real ones, using Inception-v3 features from the standard `pytorch-fid` implementation. It rewards colors that look realistic even when they differ from the original. FID is only meaningful over many images, so it is computed once on the full test set (`test_fid`) and not on the validation subsets during training. With 1,200 images it is higher than FID reported on 50,000-image benchmarks, so compare it only between runs of this project.
  - Runs finished before FID was added can get it without retraining: `python -m src.train --config <config> --eval-only` re-evaluates the best checkpoint and rewrites that run's row.

## Proposed Methods (Proposal rev01)

| Role | Method | Status |
| --- | --- | --- |
| Baseline | Simple CNN Autoencoder, MSE loss | `baseline_ae` |
| Method 1 | U-Net with ImageNet-pretrained ResNet-18 encoder, L1 loss | `unet_resnet18` |
| Method 2 | Pix2Pix (U-Net generator + PatchGAN discriminator, adversarial + L1 loss) | Not implemented yet |
| Method 3 | Palette conditional diffusion model, L2 noise loss, DDIM sampling | `palette` |

Proposal rev01 replaces Colorful Image Colorization (Zhang et al.) with Palette as Method 3 and adds Scenario 6 (DDIM sampling steps 50, 100, 250).

### Palette (Method 3)
`src/models/palette.py` frames colorization as iterative denoising of the `ab` channels conditioned on `L`:
- **Denoiser**: U-Net with residual blocks, sinusoidal timestep embedding, and self-attention at 32×32 and in the bottleneck. Input is `[L, noisy ab]` (3 channels) and output is the predicted noise (2 channels).
- **Training**: 1,000-step linear noise schedule (`β` 1e-4 to 0.02). A random timestep is sampled per image and the model minimizes the L2 loss between true and predicted noise. An EMA copy of the weights (`ema_decay: 0.999`) is used for validation, checkpoints, and testing.
- **Inference**: deterministic DDIM sampling (`ddim_eta: 0`) from Gaussian noise. `sampling_steps` sets the number of steps and defaults to 50.
- **Color space**: The original paper uses RGB. This project keeps CIELAB so the input lightness is preserved exactly and the data pipeline matches the other models.
- **Validation**: `val_loss` is the denoising loss with a fixed seed, so epochs are comparable. Validation PSNR, SSIM, and LPIPS use DDIM sampling on the first `val_metric_samples` (256) images to keep training time bounded. Test metrics, including FID, always use the full test set.

---

## Repository Structure
```text
.
├── configs/                        # YAML experiment configs (single-variable controlled)
│   ├── baseline.yaml               # Simple CNN Autoencoder (MSE, Adam, lr=2e-4, bs=16, 30ep)
│   ├── e1_lr_1e3.yaml              # E1: Learning rate ablation (lr=1e-3)
│   ├── e2_unet_resnet18.yaml       # E2: U-Net + Pretrained ResNet-18 Encoder (L1 loss)
│   ├── e3_unet_weight_decay.yaml   # E3: E2 with weight decay 1e-4
│   ├── e4_unet_bs32.yaml           # E4 (Ready): U-Net ResNet-18 with batch_size=32
│   └── m3_palette.yaml             # Method 3: Palette conditional diffusion (L2 noise loss, DDIM 50 steps)
├── data/
│   └── splits/                     # Committed reproducible manifests (train/val/test.csv)
├── notebooks/
│   ├── 01_eda_and_data_prep.ipynb          # EDA, preprocessing & split verification
│   ├── 02_baseline_and_experiments.ipynb   # Model training, curves & qualitative eval
│   └── 03_palette_colab.ipynb              # Google Colab: train Palette + Scenario 6 (Drive-backed, resumable)
├── results/
│   ├── history/                    # Per-epoch CSV logs & per-run JSON summaries
│   ├── plots/                      # EDA charts, train/val loss & metric curves
│   ├── samples/                    # Side-by-side test-set colorization grids
│   ├── metrics_summary.csv         # Consolidated results table
├── scripts/
│   └── prepare_dataset.py          # Reproducible COCO 2017 downloader & splitter
├── src/
│   ├── dataset.py                  # CIELAB PyTorch Dataset + horizontal flip augmentation
│   ├── metrics.py                  # PSNR, SSIM, LPIPS, FID evaluator
│   ├── models/
│   │   ├── baseline_ae.py          # Simple CNN Autoencoder (no skip, no pretrained)
│   │   ├── unet_resnet18.py        # U-Net with ImageNet-pretrained ResNet-18 encoder
│   │   └── palette.py              # Palette: attention U-Net denoiser + DDIM sampler
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
python -m src.train --config configs/m3_palette.yaml
```

Each epoch also writes `checkpoints/<experiment>_last.pt`. Add `--resume` to continue an interrupted run from it. To run Palette on Google Colab, open `notebooks/03_palette_colab.ipynb`; it stores the dataset, checkpoints, and results on Google Drive.

Scenario 6 reuses the trained Palette checkpoint and changes only the number of DDIM sampling steps at test time. Each run adds a row such as `m3_palette_ddim100` to `results/metrics_summary.csv`, including `test_time_sec`:
```bash
python -m src.train --config configs/m3_palette.yaml --eval-only --sampling-steps 100
python -m src.train --config configs/m3_palette.yaml --eval-only --sampling-steps 250
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
| E3 | E2 | Weight decay 0.0001 | Measure the effect of L2 regularization in Adam |
| E4 | E2 | Batch size 32 | Measure the effect of averaging gradients over more images |
| M3 | Baseline | Palette conditional diffusion, L2 noise loss, DDIM 50 steps | Benchmark Method 3 |
| E6 | M3 | DDIM sampling steps 100 and 250 (test-time only) | Measure the quality and speed trade-off of the sampler |

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
   git checkout -b exp/e3-weight-decay
   ```
2. Add a new YAML file in `configs/` changing **only one variable** per scenario.
3. Run `python -m src.train --config configs/<your_config>.yaml`.
4. Commit `configs/`, `results/history/`, `results/metrics_summary.csv`, and any new plots (never commit `data/processed_256/` or `checkpoints/`).
5. Open a Pull Request into `main`.
