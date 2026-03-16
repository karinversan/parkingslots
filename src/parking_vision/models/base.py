from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from parking_vision.data.layouts import Layout, SlotPrediction


class ParkingModel(ABC):
    @abstractmethod
    def predict_patches(self, images: List[np.ndarray]) -> List[SlotPrediction]:
        raise NotImplementedError

    @abstractmethod
    def predict_frame(self, frame: np.ndarray, layout: Layout) -> List[SlotPrediction]:
        raise NotImplementedError

    def reset_state(self) -> None:
        return None
