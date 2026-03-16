from __future__ import annotations

import json
from pathlib import Path
from typing import List

import cv2
import pandas as pd
from huggingface_hub import snapshot_download
from tqdm import tqdm

from parking_vision.data.adapters.base import DatasetAdapter
from parking_vision.data.layouts import polygon_crop


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


class PKLotAdapter(DatasetAdapter):
    def download(self) -> Path:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        if any(self.root_dir.rglob("*")):
            return self.root_dir

        source = self.cfg["dataset"]["source"]
        snapshot_download(
            repo_id=source["hf_repo_id"],
            repo_type=source.get("repo_type", "dataset"),
            local_dir=self.root_dir,
            local_dir_use_symlinks=False,
        )
        return self.root_dir

    def _resolve_image_path(self, raw_path: str) -> Path | None:
        p = Path(raw_path)
        if p.exists():
            return p

        candidates = [
            self.root_dir / raw_path,
            self.root_dir / raw_path.lstrip("/"),
            self.root_dir / Path(raw_path).name,
        ]
        for cand in candidates:
            if cand.exists():
                return cand

        name = Path(raw_path).name
        matches = list(self.root_dir.rglob(name))
        if matches:
            return matches[0]
        return None

    def _label_from_status(self, status: str | None) -> str | None:
        if status is None:
            return None

        s = str(status).strip().lower()
        occupied_aliases = set(self.cfg["dataset"]["labels"]["occupied_aliases"])
        free_aliases = set(self.cfg["dataset"]["labels"]["free_aliases"])

        if s in occupied_aliases:
            return "occupied"
        if s in free_aliases:
            return "free"

        if s in {"not occupied", "vacant"}:
            return "free"
        return None

    def _extract_weather(self, sample: dict) -> str:
        weather = sample.get("weather", {})
        if isinstance(weather, dict):
            return str(weather.get("label", "unknown")).lower()
        if isinstance(weather, str):
            return weather.lower()
        return "unknown"

    def _extract_group_key(self, sample: dict, source: str, image_path: Path) -> str:
        date_val = sample.get("date")
        if isinstance(date_val, dict) and "$date" in date_val:
            return f"{source}_{str(date_val['$date'])[:10]}"
        if isinstance(date_val, str):
            return f"{source}_{date_val[:10]}"
        return f"{source}_{image_path.stem[:10]}"

    def _to_abs_polygon(self, points, width: int, height: int):
        if not points or not isinstance(points, list):
            return None

        while isinstance(points, list) and len(points) == 1 and isinstance(points[0], list):
            child = points[0]
            if not child:
                return None
            if len(child) == 2 and all(isinstance(v, (int, float)) for v in child):
                break
            points = child

        polygon = []
        for pt in points:
            if not isinstance(pt, (list, tuple)) or len(pt) != 2:
                continue
            x, y = pt
            if 0.0 <= float(x) <= 1.0 and 0.0 <= float(y) <= 1.0:
                x = int(round(float(x) * width))
                y = int(round(float(y) * height))
            else:
                x = int(round(float(x)))
                y = int(round(float(y)))
            polygon.append([x, y])

        if len(polygon) < 3:
            return None
        return polygon

    def _build_manifest_from_fiftyone(self) -> pd.DataFrame:
        samples_path = self.root_dir / "samples.json"
        if not samples_path.exists():
            raise RuntimeError(f"samples.json not found: {samples_path}")

        with samples_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        if isinstance(payload, dict) and "samples" in payload:
            samples = payload["samples"]
        elif isinstance(payload, list):
            samples = payload
        else:
            raise RuntimeError("Unsupported samples.json format")

        patch_size = int(self.cfg["dataset"]["image"]["size"])
        patch_root = self.cache_dir / "patches"
        patch_root.mkdir(parents=True, exist_ok=True)

        rows: List[dict] = []

        for sample in tqdm(samples, desc="Building PKLot patch manifest"):
            raw_path = sample.get("filepath")
            if not raw_path:
                continue

            resolved_image = self._resolve_image_path(raw_path)
            if resolved_image is None or not resolved_image.exists():
                continue

            image = cv2.imread(str(resolved_image), cv2.IMREAD_COLOR)
            if image is None:
                continue

            h, w = image.shape[:2]
            source = str(sample.get("source", "unknown_scene"))
            weather = self._extract_weather(sample)
            group_key = self._extract_group_key(sample, source, resolved_image)

            parking_spaces = sample.get("parking_spaces", {})
            polylines = parking_spaces.get("polylines", []) if isinstance(parking_spaces, dict) else []

            for poly in polylines:
                label = self._label_from_status(poly.get("occupancy_status"))
                if label is None:
                    continue

                polygon = self._to_abs_polygon(poly.get("points"), w, h)
                if polygon is None:
                    continue

                slot_id = str(poly.get("space_id", poly.get("index", "unknown")))
                crop = polygon_crop(image, polygon, out_size=patch_size)

                out_dir = patch_root / source / weather / group_key
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{resolved_image.stem}_slot_{slot_id}_{label}.jpg"

                if not out_path.exists():
                    cv2.imwrite(str(out_path), crop)

                rows.append(
                    {
                        "image_path": str(out_path.resolve()),
                        "label": label,
                        "slot_id": slot_id,
                        "scene_id": source,
                        "group_key": group_key,
                        "weather": weather,
                        "dataset_name": "pklot",
                        "source_image_path": str(resolved_image.resolve()),
                    }
                )

        if not rows:
            raise RuntimeError("No PKLot slot patches were generated from samples.json")
        return pd.DataFrame(rows)

    def _build_manifest_from_original_tree(self) -> pd.DataFrame:
        rows: List[dict] = []

        for path in self.root_dir.rglob("*"):
            if path.suffix.lower() not in IMAGE_EXTS:
                continue

            parts = [p.lower() for p in path.parts]
            label = None
            if "occupied" in parts:
                label = "occupied"
            elif "free" in parts or "empty" in parts or "vacant" in parts:
                label = "free"

            if label is None:
                continue

            group_key = "_".join(path.parts[-5:-2]) if len(path.parts) >= 5 else path.parent.name
            scene_id = path.parts[-4] if len(path.parts) >= 4 else "unknown_scene"
            weather = next((p for p in parts if p in {"sunny", "rainy", "cloudy"}), "unknown")

            rows.append(
                {
                    "image_path": str(path.resolve()),
                    "label": label,
                    "slot_id": path.stem,
                    "scene_id": scene_id,
                    "group_key": group_key,
                    "weather": weather,
                    "dataset_name": "pklot",
                }
            )

        if not rows:
            raise RuntimeError("No PKLot images found in original-tree fallback mode")
        return pd.DataFrame(rows)

    def build_manifest(self) -> pd.DataFrame:
        if (self.root_dir / "samples.json").exists():
            return self._build_manifest_from_fiftyone()
        return self._build_manifest_from_original_tree()
