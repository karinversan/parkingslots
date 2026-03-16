from __future__ import annotations

import os
import time
from dataclasses import dataclass

import psutil


@dataclass
class ProfileSample:
    latency_ms: float
    rss_mb: float


class Profiler:
    def __init__(self) -> None:
        self.process = psutil.Process(os.getpid())

    def rss_mb(self) -> float:
        return self.process.memory_info().rss / (1024 * 1024)

    def time_call(self, fn, *args, **kwargs):
        before = self.rss_mb()
        start = time.perf_counter()
        out = fn(*args, **kwargs)
        end = time.perf_counter()
        after = self.rss_mb()
        return out, ProfileSample(latency_ms=(end - start) * 1000.0, rss_mb=max(before, after))
