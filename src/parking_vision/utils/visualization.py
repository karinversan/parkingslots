from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from parking_vision.data.layouts import Layout, SlotPrediction


STATUS_COLORS = {
    "free": (0, 200, 0),
    "occupied": (0, 0, 255),
    "unknown": (0, 180, 255),
}


def draw_layout_overlay(frame: np.ndarray, layout: Layout, predictions: List[SlotPrediction]) -> np.ndarray:
    output = frame.copy()
    pred_map = {p.slot_id: p for p in predictions}
    for slot in layout.slots:
        pred = pred_map.get(slot.slot_id)
        status = pred.status if pred else "unknown"
        color = STATUS_COLORS.get(status, STATUS_COLORS["unknown"])
        pts = np.array(slot.polygon, dtype=np.int32)
        cv2.polylines(output, [pts], isClosed=True, color=color, thickness=2)
        cv2.fillPoly(output, [pts], color=(*color[:2], color[2]))
        x, y, w, h = cv2.boundingRect(pts)
        conf = f"{pred.confidence:.2f}" if pred else "--"
        cv2.putText(output, f"{slot.slot_id}:{status[:1].upper()} {conf}", (x, max(12, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.addWeighted(output, 0.30, frame, 0.70, 0.0)


def save_confusion_matrix(cm: np.ndarray, path: str | Path) -> None:
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks([0, 1, 2], ["free", "occupied", "unknown"])
    ax.set_yticks([0, 1, 2], ["free", "occupied", "unknown"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_metric_bars(metrics_by_model: Dict[str, Dict[str, float]], path: str | Path, keys: List[str]) -> None:
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(keys))
    width = 0.35
    model_names = list(metrics_by_model.keys())
    for idx, name in enumerate(model_names):
        values = [metrics_by_model[name][k] for k in keys]
        ax.bar(x + idx * width, values, width, label=name)
    ax.set_xticks(x + width / 2, keys)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
