"""
Build the reproducible 12,000-image MS COCO 2017 colorization dataset.

Two modes:

1. Create the subset (done once, the result is committed in data/splits/):
       python scripts/prepare_dataset.py --target-count 12000 --seed 42
   - read the COCO train2017 image list,
   - shuffle it with a fixed seed and download images until 12,000 genuinely colored ones are found
     (grayscale and near-monochrome photos are skipped),
   - resize every image to 256x256 and save it to data/processed_256/,
   - split 80/10/10 into train/val/test CSV manifests and write EDA plots and statistics.

2. Reproduce the exact same images from the committed manifests (what teammates run):
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
SPLITS = ("train", "val", "test")

# One HTTP session per download thread (sessions reuse connections but are not thread-safe).
DOWNLOAD_WORKER = threading.local()


# ---------------------------------------------------------------------------
# COCO metadata
# ---------------------------------------------------------------------------

def download_file(url: str, dest: Path) -> None:
    """Download a zip archive with a progress bar, skipping it when a valid copy already exists."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and zipfile.is_zipfile(dest):
        print(f"[INFO] Found cached file: {dest}")
        return

    print(f"[INFO] Downloading {url} -> {dest} ...")
    partial = dest.with_suffix(".zip.part")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with open(partial, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as pbar:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    if not zipfile.is_zipfile(partial):
        raise RuntimeError(f"Invalid COCO archive: {partial}")
    partial.replace(dest)


def load_coco_train2017_metadata(data_dir: Path) -> List[Dict[str, Any]]:
    """Return the list of COCO train2017 images (id, file_name, width, height, ...)."""
    images_json = data_dir / "annotations" / "train2017_images.json"
    if images_json.exists():
        print(f"[INFO] Loading {images_json} ...")
        with images_json.open(encoding="utf-8") as f:
            return json.load(f)

    instances_json = data_dir / "annotations" / "instances_train2017.json"
    if not instances_json.exists():
        zip_path = data_dir / "annotations_trainval2017.zip"
        download_file(ANNOTATIONS_URL, zip_path)
        print(f"[INFO] Extracting {INSTANCES_JSON_MEMBER} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extract(INSTANCES_JSON_MEMBER, path=data_dir)

    print(f"[INFO] Loading {instances_json} ...")
    with open(instances_json, "r", encoding="utf-8") as f:
        return json.load(f)["images"]


# ---------------------------------------------------------------------------
# Downloading and filtering one image
# ---------------------------------------------------------------------------

def is_color_image(rgb: np.ndarray) -> Tuple[bool, float]:
    """
    Decide whether an RGB uint8 image has real color.

    For every pixel, `spread = max(R, G, B) - min(R, G, B)` is 0 for a pure gray pixel.
    An image counts as color when the mean spread is at least 5 and the 95th percentile is at least 12,
    which rejects grayscale and near-monochrome photos.

    Returns (is_color, mean_spread).
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        return False, 0.0
    spread = rgb.max(axis=2).astype(np.float32) - rgb.min(axis=2).astype(np.float32)
    mean_spread = float(spread.mean())
    p95_spread = float(np.percentile(spread, 95))
    return (mean_spread >= 5.0) and (p95_spread >= 12.0), mean_spread


def make_record(item: Dict[str, Any], url: str, chroma: float, width: int, height: int) -> Dict[str, Any]:
    """One row of a split CSV manifest."""
    return {
        "id": int(item["id"]),
        "file_name": item["file_name"],
        "coco_url": url,
        "orig_width": int(item.get("width", width)),
        "orig_height": int(item.get("height", height)),
        "mean_chroma": round(chroma, 3),
    }


def read_existing_image(path: Path, image_size: int) -> Optional[np.ndarray]:
    """Return the already processed RGB image at `path`, or None if it is missing or the wrong size."""
    if not path.exists():
        return None
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None or bgr.shape[0] != image_size or bgr.shape[1] != image_size:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def fetch_and_process_one(
    item: Dict[str, Any],
    out_dir: Path,
    image_size: int,
    skip_filter: bool = False,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Download one COCO image, check that it has color, resize it to image_size x image_size,
    and save it to out_dir/file_name.

    `skip_filter=True` is used when reproducing from manifests: the images are known to be valid,
    so the color check is skipped and images already on disk are not downloaded again.

    Returns (status, record) where status is 'ok', 'grayscale', or 'failed'.
    """
    file_name = item["file_name"]
    out_path = out_dir / file_name
    url = f"{COCO_ORIGIN}/train2017/{file_name}"

    if skip_filter:
        existing = read_existing_image(out_path, image_size)
        if existing is not None:
            _, chroma = is_color_image(existing)
            return "ok", make_record(item, url, chroma, image_size, image_size)

    if not hasattr(DOWNLOAD_WORKER, "session"):
        DOWNLOAD_WORKER.session = requests.Session()

    for attempt in range(3):
        try:
            response = DOWNLOAD_WORKER.session.get(url, timeout=15)
            if response.status_code != 200:
                continue
            with Image.open(io.BytesIO(response.content)) as pil_img:
                if not skip_filter and pil_img.mode not in ("RGB", "RGBA"):
                    return "grayscale", None
                rgb = np.array(pil_img.convert("RGB"))

            has_color, chroma = is_color_image(rgb)
            if not skip_filter and not has_color:
                return "grayscale", None

            orig_h, orig_w = rgb.shape[:2]
            resized = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
            saved = cv2.imwrite(
                str(out_path), cv2.cvtColor(resized, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 95]
            )
            if not saved:
                return "failed", None
            return "ok", make_record(item, url, chroma, orig_w, orig_h)
        except Exception:
            if attempt == 2:
                return "failed", None
    return "failed", None


# ---------------------------------------------------------------------------
# Mode 2: reproduce from committed manifests
# ---------------------------------------------------------------------------

def download_from_splits(args: argparse.Namespace) -> None:
    print("[INFO] Reproducing exact dataset from committed CSV manifests...")
    manifests = [pd.read_csv(args.splits_dir / f"{split}.csv") for split in SPLITS]
    items = pd.concat(manifests, ignore_index=True).to_dict(orient="records")

    ok_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(fetch_and_process_one, item, args.out_dir, args.image_size, True)
            for item in items
        ]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Downloading splits"):
            status, _ = future.result()
            ok_count += status == "ok"

    failed_count = len(items) - ok_count
    print(f"[DONE] Downloaded {ok_count}/{len(items)} images (failed: {failed_count}).")
    if failed_count:
        raise RuntimeError(f"{failed_count} split images unavailable. Rerun to retry them.")


# ---------------------------------------------------------------------------
# Mode 1: create a new subset
# ---------------------------------------------------------------------------

def collect_color_images(images: List[Dict[str, Any]], args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Walk through the shuffled COCO list in chunks until `target_count` color images are saved.

    Results are consumed in submission order, so the selected subset only depends on the seed
    (and on which downloads succeed).
    """
    records: List[Dict[str, Any]] = []
    counts = {"inspected": 0, "grayscale": 0, "failed": 0}
    cursor = 0
    max_chunk = 1500

    pbar = tqdm(total=args.target_count, desc="Collecting valid color images")
    while len(records) < args.target_count and cursor < len(images):
        remaining = args.target_count - len(records)
        # Ask for ~12% more than needed to make up for grayscale rejects and failed downloads.
        chunk_size = min(max(int(remaining * 1.12), 200), max_chunk, len(images) - cursor)
        chunk = images[cursor : cursor + chunk_size]
        cursor += chunk_size

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(fetch_and_process_one, item, args.out_dir, args.image_size, False)
                for item in chunk
            ]
            for future in futures:
                counts["inspected"] += 1
                status, record = future.result()
                if status == "ok" and record is not None:
                    if len(records) < args.target_count:
                        records.append(record)
                        pbar.update(1)
                elif status == "grayscale":
                    counts["grayscale"] += 1
                else:
                    counts["failed"] += 1
    pbar.close()

    if len(records) < args.target_count:
        raise RuntimeError(f"Only collected {len(records)} images, expected {args.target_count}.")
    return records, counts


def split_80_10_10(records: List[Dict[str, Any]], seed: int) -> Dict[str, pd.DataFrame]:
    """Sort by COCO id (deterministic), shuffle with the seed, and cut into 80% / 10% / 10%."""
    df = pd.DataFrame(records).sort_values("id").reset_index(drop=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    n_train = int(round(len(df) * 0.80))
    n_val = int(round(len(df) * 0.10))
    splits = {
        "train": df.iloc[:n_train].reset_index(drop=True),
        "val": df.iloc[n_train : n_train + n_val].reset_index(drop=True),
        "test": df.iloc[n_train + n_val :].reset_index(drop=True),
    }

    ids = {name: set(split["id"]) for name, split in splits.items()}
    assert ids["train"].isdisjoint(ids["val"])
    assert ids["train"].isdisjoint(ids["test"])
    assert ids["val"].isdisjoint(ids["test"])
    return splits


def generate_eda_plots(all_df: pd.DataFrame, out_dir: Path, plot_dir: Path, seed: int = 42) -> Dict[str, Any]:
    """Save CIELAB / resolution / chroma histograms and a sample grid, and return summary statistics."""
    plot_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # CIELAB statistics from up to 1,000 images, every 8th pixel in each direction (32x32 per image).
    sample_names = rng.choice(all_df["file_name"].values, size=min(1000, len(all_df)), replace=False)
    L_vals, a_vals, b_vals = [], [], []
    for file_name in sample_names:
        bgr = cv2.imread(str(out_dir / file_name), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        L_vals.append(lab[::8, ::8, 0].ravel())
        a_vals.append(lab[::8, ::8, 1].ravel())
        b_vals.append(lab[::8, ::8, 2].ravel())
    L_all, a_all, b_all = np.concatenate(L_vals), np.concatenate(a_vals), np.concatenate(b_vals)

    # Figure 1: distributions.
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

    # Figure 2: grayscale input vs. color target for six images.
    fig, axes = plt.subplots(2, 6, figsize=(15, 5.2))
    for col, file_name in enumerate(all_df.iloc[:6]["file_name"].tolist()):
        rgb = cv2.cvtColor(cv2.imread(str(out_dir / file_name), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        lab = cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
        axes[0, col].imshow(lab[:, :, 0] / 100.0, cmap="gray", vmin=0, vmax=1)
        axes[1, col].imshow(rgb)
        axes[0, col].axis("off")
        axes[1, col].axis("off")
    axes[0, 0].set_title("Input (L)", fontsize=10)
    axes[1, 0].set_title("Target (RGB)", fontsize=10)
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


def create_subset(args: argparse.Namespace) -> None:
    images = load_coco_train2017_metadata(args.data_dir)
    print(f"[INFO] Total COCO 2017 train images in metadata: {len(images):,}")

    images.sort(key=lambda image: image["id"])
    random.Random(args.seed).shuffle(images)

    records, counts = collect_color_images(images, args)
    splits = split_80_10_10(records, args.seed)
    for name, split in splits.items():
        split.to_csv(args.splits_dir / f"{name}.csv", index=False)

    print("[INFO] Generating EDA plots...")
    all_df = pd.concat(splits.values(), ignore_index=True)
    eda_stats = generate_eda_plots(all_df, args.out_dir, args.plot_dir, seed=args.seed)

    sizes = {name: len(split) for name, split in splits.items()}
    stats = {
        "source": "MS COCO 2017 (train2017)",
        "total_pool_size": len(images),
        "images_inspected": counts["inspected"],
        "grayscale_filtered_out": counts["grayscale"],
        "failed_downloads": counts["failed"],
        "final_dataset_size": sum(sizes.values()),
        "image_size": args.image_size,
        "seed": args.seed,
        "splits": sizes,
        "eda": eda_stats,
    }
    stats_path = args.splits_dir / "dataset_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"[DONE] Saved splits to {args.splits_dir} (train={sizes['train']}, val={sizes['val']}, test={sizes['test']})")
    print(f"[DONE] Filtered out {counts['grayscale']} grayscale images; stats saved to {stats_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the MS COCO 2017 12k colorization dataset.")
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
        help="Download the exact images listed in the existing data/splits/{train,val,test}.csv",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.splits_dir.mkdir(parents=True, exist_ok=True)

    if args.from_splits:
        download_from_splits(args)
    else:
        create_subset(args)


if __name__ == "__main__":
    main()
