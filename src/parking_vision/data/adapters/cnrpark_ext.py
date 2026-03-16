from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

from parking_vision.data.adapters.base import DatasetAdapter

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


class CNRParkEXTAdapter(DatasetAdapter):
    def download(self) -> Path:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        if any(self.root_dir.rglob("*")):
            return self.root_dir

        url = self.cfg["dataset"]["source"].get("url", "")
        if not url:
            raise RuntimeError(
                "CNRPark+EXT adapter is ready, but no auto-download URL was provided. "
                "Set dataset.source.url in configs/dataset_cnrpark_ext.yaml or place the dataset into data/cnrpark_ext."
            )

        archive_path = self.root_dir.parent / Path(url).name
        urlretrieve(url, archive_path)
        if archive_path.suffix == ".zip":
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(self.root_dir)
        else:
            with tarfile.open(archive_path, "r:*") as tf:
                tf.extractall(self.root_dir)
        return self.root_dir

    def build_manifest(self) -> pd.DataFrame:
        rows = []
        for path in self.root_dir.rglob("*"):
            if path.suffix.lower() not in IMAGE_EXTS:
                continue
            parts = [p.lower() for p in path.parts]
            if any(alias in parts for alias in self.cfg["dataset"]["labels"]["occupied_aliases"]):
                label = "occupied"
            elif any(alias in parts for alias in self.cfg["dataset"]["labels"]["free_aliases"]):
                label = "free"
            else:
                continue
            rows.append(
                {
                    "image_path": str(path.resolve()),
                    "label": label,
                    "slot_id": path.stem,
                    "scene_id": path.parent.name,
                    "group_key": path.parent.name,
                    "weather": "unknown",
                    "dataset_name": "cnrpark_ext",
                }
            )
        if not rows:
            raise RuntimeError("No CNRPark+EXT image patches found. Check the dataset root and folder names.")
        return pd.DataFrame(rows)
