from __future__ import annotations

from pathlib import Path

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset

from parking_vision.data.augmentations import build_eval_augmentations, build_train_augmentations
from parking_vision.utils.metrics import LABEL_TO_INT


class ParkingPatchDataset(Dataset):
    def __init__(self, manifest_path: str | Path, image_size: int = 224, train: bool = False):
        self.df = pd.read_csv(manifest_path)
        self.image_size = image_size
        self.transform = build_train_augmentations(image_size) if train else build_eval_augmentations(image_size)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = cv2.imread(row["image_path"], cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {row['image_path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed = self.transform(image=image)
        target = LABEL_TO_INT[row["label"]]
        return {
            "image": transformed["image"],
            "target": torch.tensor(target, dtype=torch.long),
            "path": row["image_path"],
            "slot_id": row.get("slot_id", str(idx)),
        }
