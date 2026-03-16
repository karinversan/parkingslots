from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import cv2
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from parking_vision.data.layouts import Layout, SlotPrediction, polygon_crop
from parking_vision.models.base import ParkingModel


def shannon_entropy(gray: np.ndarray) -> float:
    hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).flatten()
    hist /= max(hist.sum(), 1e-6)
    hist = hist[hist > 0]
    return float(-(hist * np.log2(hist)).sum())


def basic_shadow_suppression(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.equalizeHist(v)
    return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)


class HandcraftedFeatureExtractor:
    def __init__(self, image_size: int = 160):
        self.image_size = image_size

    def compute(self, image: np.ndarray, reference: np.ndarray | None = None) -> np.ndarray:
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = basic_shadow_suppression(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        sobelx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobelx ** 2 + sobely ** 2)
        edges = cv2.Canny(blur, 80, 160)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        feats = [
            float(gray.mean()),
            float(gray.std()),
            float(np.median(gray)),
            float(grad_mag.mean()),
            float(grad_mag.std()),
            float((edges > 0).mean()),
            float(cv2.Laplacian(blur, cv2.CV_32F).var()),
            float(shannon_entropy(gray)),
            float(s.mean()),
            float(v.mean()),
            float(v.std()),
        ]

        if reference is not None:
            reference = cv2.resize(reference, (self.image_size, self.image_size))
            reference = basic_shadow_suppression(reference)
            ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
            delta = cv2.absdiff(gray, ref_gray)
            delta_ratio = float((delta > 25).mean())
            feats.extend(
                [
                    float(delta.mean()),
                    float(delta.std()),
                    delta_ratio,
                ]
            )
        else:
            feats.extend([0.0, 0.0, 0.0])

        return np.asarray(feats, dtype=np.float32)


@dataclass
class ClassicThresholdState:
    current: str = "unknown"
    stable_count: int = 0


class ClassicFastModel(ParkingModel):
    def __init__(self, cfg: dict, artifact_path: str | None = None):
        self.cfg = cfg
        self.extractor = HandcraftedFeatureExtractor(image_size=cfg["data"].get("image_size", 160))
        self.unknown_threshold = cfg["model"].get("unknown_threshold", 0.15)
        self.occupied_threshold = cfg["model"].get("occupied_threshold", 0.5)
        self.temporal_cfg = cfg["model"].get("temporal", {})
        self.use_classifier = cfg["model"].get("use_logistic_regression", True)
        self.classifier: LogisticRegression | None = None
        self.scaler: StandardScaler | None = None
        self.references: Dict[str, np.ndarray] = {}
        self.states: Dict[str, ClassicThresholdState] = defaultdict(ClassicThresholdState)
        if artifact_path and Path(artifact_path).exists():
            payload = joblib.load(artifact_path)
            self.classifier = payload.get("classifier")
            self.scaler = payload.get("scaler")
            self.references = payload.get("references", {})
            self.unknown_threshold = float(payload.get("unknown_threshold", self.unknown_threshold))
            self.occupied_threshold = float(payload.get("occupied_threshold", self.occupied_threshold))

    def calibrate_references(self, slot_crops: Dict[str, List[np.ndarray]]) -> None:
        for slot_id, crops in slot_crops.items():
            if not crops:
                continue
            brightness = np.array([cv2.cvtColor(c, cv2.COLOR_BGR2GRAY).mean() for c in crops])
            ref_idx = int(np.argmin(np.abs(brightness - np.quantile(brightness, 0.35))))
            self.references[slot_id] = crops[ref_idx]

    def _predict_single(self, slot_id: str, image: np.ndarray, apply_temporal: bool = True) -> SlotPrediction:
        ref = self.references.get(slot_id)
        feats = self.extractor.compute(image, reference=ref).reshape(1, -1)
        model_input = self.scaler.transform(feats) if self.scaler is not None else feats

        if self.classifier is not None:
            probs = self.classifier.predict_proba(model_input)[0]
            occupied_prob = float(probs[1])
        else:
            delta_energy = float(feats[0, -3])
            edge_density = float(feats[0, 5])
            occupied_prob = float(np.clip(0.35 * edge_density + 0.65 * (delta_energy / 50.0), 0.0, 1.0))

        confidence = max(occupied_prob, 1.0 - occupied_prob)
        if abs(occupied_prob - 0.5) < self.unknown_threshold:
            label = "unknown"
        elif not apply_temporal:
            label = "occupied" if occupied_prob >= self.occupied_threshold else "free"
        else:
            enter_th = self.temporal_cfg.get("occupied_enter_threshold", 0.58)
            exit_th = self.temporal_cfg.get("occupied_exit_threshold", 0.42)
            stable_frames = self.temporal_cfg.get("stable_frames", 4)
            state = self.states[slot_id]

            proposed = "occupied" if occupied_prob >= self.occupied_threshold else "free"
            if state.current == "occupied" and occupied_prob < exit_th:
                proposed = "free"
            if state.current in {"free", "unknown"} and occupied_prob > enter_th:
                proposed = "occupied"

            if proposed == state.current:
                state.stable_count += 1
            else:
                state.stable_count = 1

            if state.stable_count >= stable_frames or state.current == "unknown":
                state.current = proposed
            label = state.current
        return SlotPrediction(slot_id=slot_id, status=label, confidence=float(confidence), logits=[1-occupied_prob, occupied_prob])

    def predict_patches(self, images: List[np.ndarray]) -> List[SlotPrediction]:
        preds = []
        for idx, image in enumerate(images):
            preds.append(self._predict_single(str(idx), image, apply_temporal=False))
        return preds

    def predict_frame(self, frame: np.ndarray, layout: Layout) -> List[SlotPrediction]:
        preds = []
        for slot in layout.slots:
            crop = polygon_crop(frame, slot.polygon, out_size=self.extractor.image_size)
            preds.append(self._predict_single(slot.slot_id, crop, apply_temporal=True))
        return preds

    def reset_state(self) -> None:
        self.states.clear()
