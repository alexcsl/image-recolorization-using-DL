"""
Reproducible MS COCO 2017 (12,000-image subset) Downloader & Preprocessor.

Usage:
  # 1. Create the 12,000-image subset + 80/10/10 splits from official COCO 2017 train2017 metadata:
  python scripts/prepare_dataset.py --target-count 12000 --seed 42

  # 2. Teammate mode: reproduce the EXACT same 12,000 images from committed CSV manifests:
  python scripts/prepare_dataset.py --from-splits
"""

import argparse
import concurrent.futures
import io
import json
import random
import threading
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm

COCO_ORIGIN = "https://s3.amazonaws.com/images.cocodataset.org"
ANNOTATIONS_URL = f"{COCO_ORIGIN}/annotations/annotations_trainval2017.zip"
INSTANCES_JSON_MEMBER = "annotations/instances_train2017.json"
DOWNLOAD_WORKER = threading.local()


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and zipfile.is_zipfile(dest):
        print(f"[INFO] Found cached file: {dest}")
        return
    print(f"[INFO] Downloading {url} -> {dest} ...")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        partial = dest.with_suffix(".zip.part")
        with open(partial, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name
        ) as pbar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    if not zipfile.is_zipfile(partial):
        raise RuntimeError(f"Invalid COCO archive: {partial}")
    partial.replace(dest)


def load_coco_train2017_metadata(data_dir: Path) -> List[Dict[str, Any]]:
    json_path = data_dir / "annotations" / "instances_train2017.json"
    if not json_path.exists():
        zip_path = data_dir / "annotations_trainval2017.zip"
        download_file(ANNOTATIONS_URL, zip_path)
        print(f"[INFO] Extracting {INSTANCES_JSON_MEMBER} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extract(INSTANCES_JSON_MEMBER, path=data_dir)
    print(f"[INFO] Loading {json_path} ...")
    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return meta["images"]


def is_color_image(rgb: np.ndarray) -> Tuple[bool, float]:
    """
    Detect whether an RGB uint8 image has genuine color (not grayscale / monochrome).
    Returns (is_color, mean_chroma_spread).
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        return False, 0.0
    spread = rgb.max(axis=2).astype(np.float32) - rgb.min(axis=2).astype(np.float32)
    mean_spread = float(spread.mean())
    p95_spread = float(np.percentile(spread, 95))
    # Thresholds reject pure grayscale and near-monochrome / sepia-less B&W photos
    is_color = (mean_spread >= 5.0) and (p95_spread >= 12.0)
    return is_color, mean_spread


def fetch_and_process_one(
    item: Dict[str, Any],
    out_dir: Path,
    image_size: int,
    skip_filter: bool = False,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Download one COCO image, verify it is genuinely color, resize to image_size x image_size,
    and save to out_dir/file_name.

    Returns status in {'ok', 'grayscale', 'failed'} and record dict if 'ok'.
    """
    file_name = item["file_name"]
    out_path = out_dir / file_name
    url = f"{COCO_ORIGIN}/train2017/{file_name}"

    if skip_filter and out_path.exists():
        bgr = cv2.imread(str(out_path), cv2.IMREAD_COLOR)
        if bgr is not None and bgr.shape[0] == image_size and bgr.shape[1] == image_size:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            _, chroma = is_color_image(rgb)
            return "ok", {
                "id": int(item["id"]),
                "file_name": file_name,
                "coco_url": url,
                "orig_width": int(item.get("width", image_size)),
                "orig_height": int(item.get("height", image_size)),
                "mean_chroma": round(chroma, 3),
            }

    if not hasattr(DOWNLOAD_WORKER, "session"):
        DOWNLOAD_WORKER.session = requests.Session()
    for attempt in range(3):
        try:
            resp = DOWNLOAD_WORKER.session.get(url, timeout=15)
            if resp.status_code != 200:
                continue
            with Image.open(io.BytesIO(resp.content)) as pil_img:
                if not skip_filter and pil_img.mode not in ("RGB", "RGBA"):
                    return "grayscale", None
                rgb = np.array(pil_img.convert("RGB"))
            if not skip_filter:
                ok_color, chroma = is_color_image(rgb)
                if not ok_color:
                    return "grayscale", None
            else:
                _, chroma = is_color_image(rgb)

            orig_h, orig_w = rgb.shape[:2]
            resized = cv2.resize(
                rgb, (image_size, image_size), interpolation=cv2.INTER_AREA
            )
            bgr_out = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
            if not cv2.imwrite(str(out_path), bgr_out, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
                return "failed", None
            return "ok", {
                "id": int(item["id"]),
                "file_name": file_name,
                "coco_url": url,
                "orig_width": int(item.get("width", orig_w)),
                "orig_height": int(item.get("height", orig_h)),
                "mean_chroma": round(chroma, 3),
            }
        except Exception:
            if attempt == 2:
                return "failed", None
    return "failed", None


def generate_eda_plots(
    all_df: pd.DataFrame,
    out_dir: Path,
    plot_dir: Path,
    seed: int = 42,
) -> Dict[str, Any]:
    """Generate EDA charts (resolution distribution, CIELAB histograms, sample grid)."""
    plot_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # Sample up to 1,000 images for fast CIELAB distribution statistics
    sample_names = rng.choice(
        all_df["file_name"].values, size=min(1000, len(all_df)), replace=False
    )
    L_vals, a_vals, b_vals = [], [], []
    for fname in sample_names:
        bgr = cv2.imread(str(out_dir / fname), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        # Subsample pixels (stride 8 -> 32x32=1024 pixels/image)
        L_vals.append(lab[::8, ::8, 0].ravel())
        a_vals.append(lab[::8, ::8, 1].ravel())
        b_vals.append(lab[::8, ::8, 2].ravel())

    L_all = np.concatenate(L_vals)
    a_all = np.concatenate(a_vals)
    b_all = np.concatenate(b_vals)

    # 1. CIELAB + Resolution EDA figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].hist(L_all, bins=50, color="#4A5568", alpha=0.85)
    axes[0, 0].set_title("CIELAB L (Lightness) Distribution")
    axes[0, 0].set_xlabel("L [0, 100]")
    axes[0, 0].set_ylabel("Pixel Count (sampled)")

    axes[0, 1].hist(a_all, bins=60, color="#E53E3E", alpha=0.65, label="a* (Green-Red)")
    axes[0, 1].hist(b_all, bins=60, color="#3182CE", alpha=0.65, label="b* (Blue-Yellow)")
    axes[0, 1].set_title("CIELAB Chrominance (a*, b*) Distribution")
    axes[0, 1].set_xlabel("Value [-128, 127]")
    axes[0, 1].legend()

    axes[1, 0].hist(all_df["orig_width"], bins=30, alpha=0.65, label="Original Width")
    axes[1, 0].hist(all_df["orig_height"], bins=30, alpha=0.65, label="Original Height")
    axes[1, 0].set_title("Original COCO Image Resolution Distribution")
    axes[1, 0].set_xlabel("Pixels")
    axes[1, 0].set_ylabel("Image Count")
    axes[1, 0].legend()

    axes[1, 1].hist(all_df["mean_chroma"], bins=40, color="#38A169", alpha=0.85)
    axes[1, 1].set_title("Image Mean Chroma Spread (Post Grayscale Filter)")
    axes[1, 1].set_xlabel("Mean(max(R,G,B) - min(R,G,B))")
    axes[1, 1].set_ylabel("Image Count")

    plt.tight_layout()
    eda_path = plot_dir / "dataset_eda.png"
    plt.savefig(eda_path, dpi=150)
    plt.close(fig)

    # 2. Sample pairs (Grayscale L vs Ground Truth RGB)
    fig, axes = plt.subplots(2, 6, figsize=(15, 5.2))
    sample_6 = all_df.iloc[:6]["file_name"].tolist()
    for col, fname in enumerate(sample_6):
        bgr = cv2.imread(str(out_dir / fname), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        lab = cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
        L_img = lab[:, :, 0] / 100.0
        axes[0, col].imshow(L_img, cmap="gray", vmin=0, vmax=1)
        axes[0, col].axis("off")
        if col == 0:
            axes[0, col].set_title("Input (L)", fontsize=10)
        axes[1, col].imshow(rgb)
        axes[1, col].axis("off")
        if col == 0:
            axes[1, col].set_title("Target (RGB)", fontsize=10)
    plt.suptitle("MS COCO 2017 Preprocessed Samples (256x256): Grayscale Input vs Color Target", fontsize=12)
    plt.tight_layout()
    samples_path = plot_dir / "dataset_samples.png"
    plt.savefig(samples_path, dpi=150)
    plt.close(fig)

    return {
        "L_mean": round(float(L_all.mean()), 2),
        "L_std": round(float(L_all.std()), 2),
        "a_mean": round(float(a_all.mean()), 2),
        "a_std": round(float(a_all.std()), 2),
        "b_mean": round(float(b_all.mean()), 2),
        "b_std": round(float(b_all.std()), 2),
        "orig_width_median": int(all_df["orig_width"].median()),
        "orig_height_median": int(all_df["orig_height"].median()),
        "eda_plot": str(eda_path),
        "samples_plot": str(samples_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MS COCO 2017 12k colorization dataset.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed_256"))
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--plot-dir", type=Path, default=Path("results/plots"))
    parser.add_argument("--target-count", type=int, default=12000)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument(
        "--from-splits",
        action="store_true",
        help="Download exact images listed in existing data/splits/{train,val,test}.csv",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.splits_dir.mkdir(parents=True, exist_ok=True)

    if args.from_splits:
        print("[INFO] Reproducing exact dataset from committed CSV manifests...")
        dfs = [pd.read_csv(args.splits_dir / f"{s}.csv") for s in ("train", "val", "test")]
        all_items = pd.concat(dfs, ignore_index=True).to_dict(orient="records")
        ok_count = 0
        failed_count = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [
                ex.submit(fetch_and_process_one, item, args.out_dir, args.image_size, True)
                for item in all_items
            ]
            for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Downloading splits"):
                status, _ = fut.result()
                if status == "ok":
                    ok_count += 1
                else:
                    failed_count += 1
        print(f"[DONE] Downloaded {ok_count}/{len(all_items)} images (failed: {failed_count}).")
        if failed_count:
            raise RuntimeError(f"{failed_count} split images unavailable. Rerun to retry them.")
        return

    images = load_coco_train2017_metadata(args.data_dir)
    print(f"[INFO] Total COCO 2017 train images in metadata: {len(images):,}")

    rng = random.Random(args.seed)
    images.sort(key=lambda image: image["id"])
    rng.shuffle(images)

    # Fetch in chunks until we reach exact target_count valid color images
    records: List[Dict[str, Any]] = []
    grayscale_rejected = 0
    failed_downloads = 0
    inspected = 0
    cursor = 0
    chunk_size = 1500

    pbar = tqdm(total=args.target_count, desc="Collecting valid color images")
    while len(records) < args.target_count and cursor < len(images):
        remaining = args.target_count - len(records)
        # Oversample slightly by 12% to account for grayscale filtering & rare network blips
        batch_n = min(max(int(remaining * 1.12), 200), chunk_size, len(images) - cursor)
        batch = images[cursor : cursor + batch_n]
        cursor += batch_n

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [
                ex.submit(fetch_and_process_one, item, args.out_dir, args.image_size, False)
                for item in batch
            ]
            for fut in futures:
                inspected += 1
                status, rec = fut.result()
                if status == "ok" and rec is not None:
                    if len(records) < args.target_count:
                        records.append(rec)
                        pbar.update(1)
                elif status == "grayscale":
                    grayscale_rejected += 1
                else:
                    failed_downloads += 1
    pbar.close()

    if len(records) < args.target_count:
        raise RuntimeError(f"Only collected {len(records)} images, expected {args.target_count}.")

    # Sort deterministically by COCO image ID, then shuffle with fixed seed for split
    df = pd.DataFrame(records).sort_values("id").reset_index(drop=True)
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    n_total = len(df)
    n_train = int(round(n_total * 0.80))
    n_val = int(round(n_total * 0.10))
    n_test = n_total - n_train - n_val

    train_df = df.iloc[:n_train].reset_index(drop=True)
    val_df = df.iloc[n_train : n_train + n_val].reset_index(drop=True)
    test_df = df.iloc[n_train + n_val :].reset_index(drop=True)

    # Verify zero overlap
    assert set(train_df["id"]).isdisjoint(set(val_df["id"]))
    assert set(train_df["id"]).isdisjoint(set(test_df["id"]))
    assert set(val_df["id"]).isdisjoint(set(test_df["id"]))

    train_df.to_csv(args.splits_dir / "train.csv", index=False)
    val_df.to_csv(args.splits_dir / "val.csv", index=False)
    test_df.to_csv(args.splits_dir / "test.csv", index=False)

    print("[INFO] Generating EDA plots...")
    eda_stats = generate_eda_plots(df, args.out_dir, args.plot_dir, seed=args.seed)

    stats = {
        "source": "MS COCO 2017 (train2017)",
        "total_pool_size": len(images),
        "images_inspected": inspected,
        "grayscale_filtered_out": grayscale_rejected,
        "failed_downloads": failed_downloads,
        "final_dataset_size": n_total,
        "image_size": args.image_size,
        "seed": args.seed,
        "splits": {
            "train": n_train,
            "val": n_val,
            "test": n_test,
        },
        "eda": eda_stats,
    }
    stats_path = args.splits_dir / "dataset_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"[DONE] Saved splits to {args.splits_dir} (train={n_train}, val={n_val}, test={n_test})")
    print(f"[DONE] Filtered out {grayscale_rejected} grayscale images; stats saved to {stats_path}")


if __name__ == "__main__":
    main()
