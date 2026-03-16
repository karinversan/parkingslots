from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from parking_vision.config import load_yaml
from parking_vision.models.factory import build_model


@dataclass
class AppState:
    config_path: str
    cfg: dict = field(init=False)
    models: dict = field(default_factory=dict)

    def __post_init__(self):
        self.cfg = load_yaml(self.config_path)

    def _resolve_path(self, raw_path: str | None) -> str | None:
        if raw_path is None:
            return None
        path = Path(raw_path)
        if path.is_absolute():
            return str(path)
        return str((Path(self.config_path).resolve().parents[1] / path).resolve())

    def get_model(self, model_key: str):
        if model_key in self.models:
            return self.models[model_key]
        model_cfg = self.cfg["models"][model_key]
        artifact = model_cfg.get("checkpoint") or model_cfg.get("artifact")
        self.models[model_key] = build_model(
            model_type=model_cfg["type"],
            config_path=self._resolve_path(model_cfg["config"]),
            artifact_path=self._resolve_path(artifact),
        )
        return self.models[model_key]
