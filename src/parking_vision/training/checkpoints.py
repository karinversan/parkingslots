from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(payload: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
