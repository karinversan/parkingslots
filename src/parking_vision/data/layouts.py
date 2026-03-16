from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import cv2
import numpy as np

from parking_vision.utils.io import load_json, save_json


@dataclass
class Slot:
    slot_id: str
    polygon: List[List[int]]


@dataclass
class Layout:
    camera_id: str
    image_width: int
    image_height: int
    slots: List[Slot]


@dataclass
class SlotPrediction:
    slot_id: str
    status: str
    confidence: float
    logits: List[float] | None = None


def load_layout(path: str | Path) -> Layout:
    payload = load_json(path)
    slots = [Slot(slot_id=s["slot_id"], polygon=s["polygon"]) for s in payload["slots"]]
    return Layout(
        camera_id=payload["camera_id"],
        image_width=payload["image_width"],
        image_height=payload["image_height"],
        slots=slots,
    )


def save_layout(layout: Layout, path: str | Path) -> None:
    payload = {
        "camera_id": layout.camera_id,
        "image_width": layout.image_width,
        "image_height": layout.image_height,
        "slots": [{"slot_id": s.slot_id, "polygon": s.polygon} for s in layout.slots],
    }
    save_json(payload, path)


def polygon_crop(image: np.ndarray, polygon: Sequence[Sequence[int]], out_size: int = 224) -> np.ndarray:
    pts = np.array(polygon, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(pts)
    roi = image[y:y + h, x:x + w].copy()
    shifted = pts - np.array([[x, y]])
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [shifted], 255)
    masked = cv2.bitwise_and(roi, roi, mask=mask)
    resized = cv2.resize(masked, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    return resized
