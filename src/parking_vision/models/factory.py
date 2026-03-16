from __future__ import annotations

from pathlib import Path

from parking_vision.config import load_yaml
from parking_vision.models.model_a_mobilenet import ModelAInference
from parking_vision.models.model_b_classic import ClassicFastModel


def build_model(model_type: str, config_path: str, artifact_path: str | None = None):
    cfg = load_yaml(config_path)
    model_type = model_type.lower()
    if model_type == "model_a":
        if artifact_path is None:
            raise ValueError("Model A requires a checkpoint path.")
        return ModelAInference(checkpoint_path=artifact_path, cfg=cfg)
    if model_type == "model_b":
        return ClassicFastModel(cfg=cfg, artifact_path=artifact_path)
    raise ValueError(f"Unknown model type: {model_type}")
