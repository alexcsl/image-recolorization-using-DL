"""PyTorch dataset that turns color photos into (L input, ab target) pairs."""

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.utils import rgb_to_lab_normalized


class ColorizationDataset(Dataset):
    """
    Dataset for Black-and-White Image Colorization in CIELAB space.

    Reads RGB images listed in a split CSV manifest, optionally applies random
    horizontal flip (training only), resizes to `image_size x image_size` if
    needed, and returns normalized tensors:
      - L:  (1, H, W) float32 in [-1, 1]
      - ab: (2, H, W) float32 in [-1, 1]
      - file_name: str
    """

    def __init__(
        self,
        csv_path: str | Path,
        image_dir: str | Path,
        image_size: int = 256,
        augment: bool = False,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.augment = augment

        self.df = pd.read_csv(self.csv_path)
        if "file_name" not in self.df.columns:
            raise ValueError(f"'file_name' column missing in {self.csv_path}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        file_name = str(self.df.iloc[idx]["file_name"])
        img_path = self.image_dir / file_name

        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if rgb.shape[0] != self.image_size or rgb.shape[1] != self.image_size:
            rgb = cv2.resize(
                rgb,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_AREA,
            )

        if self.augment and np.random.rand() < 0.5:
            rgb = np.ascontiguousarray(rgb[:, ::-1, :])

        L_norm, ab_norm = rgb_to_lab_normalized(rgb)
        L_tensor = torch.from_numpy(L_norm).unsqueeze(0)  # (1, H, W)
        ab_tensor = torch.from_numpy(ab_norm).permute(2, 0, 1)  # (2, H, W)
        return L_tensor, ab_tensor, file_name
