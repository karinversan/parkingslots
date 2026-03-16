from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

from parking_vision.data.layouts import Layout, SlotPrediction, polygon_crop
from parking_vision.models.base import ParkingModel
from parking_vision.models.smoothing import TemporalConfig, TemporalStateFilter
from parking_vision.utils.metrics import INT_TO_LABEL


class SlotClassifierModelA(nn.Module):
    def __init__(
        self,
        backbone: str = "mobilenet_v3_large",
        num_classes: int = 3,
        pretrained: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        weights = None
        if pretrained:
            if backbone == "mobilenet_v3_small":
                weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
            else:
                weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2

        if backbone == "mobilenet_v3_small":
            net = models.mobilenet_v3_small(weights=weights)
        else:
            net = models.mobilenet_v3_large(weights=weights)

        in_features = net.classifier[-1].in_features
        net.classifier[-1] = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )
        self.net = net

    def forward(self, x):
        logits = self.net(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits


@dataclass
class ModelAConfig:
    image_size: int = 224
    unknown_threshold: float = 0.55
    device: str = "cpu"
    smoothing: TemporalConfig = field(default_factory=TemporalConfig)


class ModelAInference(ParkingModel):
    def __init__(self, checkpoint_path: str, cfg: dict):
        device_cfg = str(cfg.get("device", "cpu")).lower()
        if device_cfg == "mps" and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif device_cfg == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        backbone = cfg["model"].get("backbone", "mobilenet_v3_large")
        dropout = cfg["model"].get("dropout", 0.2)
        self.image_size = cfg["data"].get("image_size", 224)
        self.unknown_threshold = cfg["model"].get("unknown_threshold", 0.55)

        self.model = SlotClassifierModelA(
            backbone=backbone,
            num_classes=cfg["model"].get("num_classes", 3),
            pretrained=False,
            dropout=dropout,
        )
        state = torch.load(checkpoint_path, map_location="cpu")
        if "model_state" in state:
            self.model.load_state_dict(state["model_state"])
        else:
            self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        self.smoother = TemporalStateFilter(
            TemporalConfig(**cfg.get("inference", {}).get("smoothing", {}))
        )

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image = (image - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
            [0.229, 0.224, 0.225], dtype=np.float32
        )
        tensor = torch.from_numpy(image).permute(2, 0, 1)
        return tensor

    @torch.inference_mode()
    def predict_patches(self, images: List[np.ndarray]) -> List[SlotPrediction]:
        if not images:
            return []
        batch = torch.stack([self._preprocess(img) for img in images]).to(self.device)
        logits = self.model(batch)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = []
        for i, p in enumerate(probs):
            pred_idx = int(p.argmax())
            confidence = float(p[pred_idx])
            status = INT_TO_LABEL[pred_idx]
            if confidence < self.unknown_threshold:
                status = "unknown"
            preds.append(SlotPrediction(slot_id=str(i), status=status, confidence=confidence, logits=p.tolist()))
        return preds

    def predict_frame(self, frame: np.ndarray, layout: Layout) -> List[SlotPrediction]:
        patches = [polygon_crop(frame, slot.polygon, out_size=self.image_size) for slot in layout.slots]
        raw_preds = self.predict_patches(patches)
        results = []
        for slot, pred in zip(layout.slots, raw_preds):
            occupied_prob = pred.logits[1] if pred.logits is not None and len(pred.logits) > 1 else pred.confidence
            smooth_status = self.smoother.update(slot.slot_id, float(occupied_prob))
            results.append(
                SlotPrediction(
                    slot_id=slot.slot_id,
                    status=smooth_status if pred.status != "unknown" else "unknown",
                    confidence=pred.confidence,
                    logits=pred.logits,
                )
            )
        return results

    def reset_state(self) -> None:
        self.smoother.reset()
