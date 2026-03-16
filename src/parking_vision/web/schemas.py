from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class StreamRequest(BaseModel):
    rtsp_url: str
    layout_path: str
    model_key: str = "strong_baseline"
    max_frames: int = 120
    stride: int = 5


class DemoRequest(BaseModel):
    demo_id: str
    model_key: str = "strong_baseline"
    max_frames: int = 120
    stride: int = 5


class SlotStatusResponse(BaseModel):
    slot_id: str
    status: str
    confidence: float


class PredictionResponse(BaseModel):
    model_key: str
    total_slots: int
    occupied: int
    free: int
    unknown: int
    occupancy_rate: float
    latency_ms: float
    rss_mb: float
    output_path: Optional[str] = None
    slots: List[SlotStatusResponse]
