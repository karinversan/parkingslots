from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass


@dataclass
class TemporalConfig:
    window_size: int = 5
    occupied_enter_threshold: float = 0.65
    occupied_exit_threshold: float = 0.45


class TemporalStateFilter:
    def __init__(self, config: TemporalConfig):
        self.config = config
        self.scores = defaultdict(lambda: deque(maxlen=config.window_size))
        self.state = {}

    def update(self, slot_id: str, occupied_prob: float) -> str:
        self.scores[slot_id].append(occupied_prob)
        avg = sum(self.scores[slot_id]) / len(self.scores[slot_id])

        current = self.state.get(slot_id, "unknown")
        if current in {"free", "unknown"} and avg >= self.config.occupied_enter_threshold:
            current = "occupied"
        elif current == "occupied" and avg <= self.config.occupied_exit_threshold:
            current = "free"
        elif current == "unknown":
            current = "occupied" if avg >= 0.5 else "free"

        self.state[slot_id] = current
        return current

    def reset(self) -> None:
        self.scores.clear()
        self.state.clear()
