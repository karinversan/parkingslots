from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib


class FeatureCache:
    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, key: str) -> Path:
        safe = key.replace("/", "_").replace("\\", "_").replace(":", "_")
        return self.cache_dir / f"{safe}.joblib"

    def get(self, key: str) -> Any | None:
        path = self.path_for(key)
        if not path.exists():
            return None
        return joblib.load(path)

    def set(self, key: str, value: Any) -> None:
        joblib.dump(value, self.path_for(key))
