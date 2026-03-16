from __future__ import annotations

from pathlib import Path
from typing import Generator, Iterable, Tuple

import cv2
import imageio.v2 as imageio
import numpy as np


def iter_video_frames(source: str, max_frames: int | None = None, stride: int = 1) -> Generator[Tuple[int, np.ndarray], None, None]:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video/stream source: {source}")

    idx = 0
    emitted = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            yield idx, frame
            emitted += 1
            if max_frames is not None and emitted >= max_frames:
                break
        idx += 1
    cap.release()


def write_video(frames: Iterable[np.ndarray], output_path: str | Path, fps: float = 12.0) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(output_path, fps=fps, codec="libx264") as writer:
        for frame in frames:
            writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
