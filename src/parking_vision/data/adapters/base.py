from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class DatasetAdapter(ABC):
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.root_dir = Path(cfg["dataset"]["root_dir"])
        self.cache_dir = Path(cfg["dataset"]["cache_dir"])

    @abstractmethod
    def download(self) -> Path:
        raise NotImplementedError

    @abstractmethod
    def build_manifest(self) -> pd.DataFrame:
        raise NotImplementedError
