from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


TargetClass = Literal[
    "person",
    "rider",
    "car",
    "bus",
    "truck",
    "bike",
    "motorcycle",
    "traffic light",
    "traffic sign",
    "train",
]


class Detection(BaseModel):
    label: TargetClass
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: List[float] = Field(
        ..., description="[x1, y1, x2, y2] in pixel coordinates"
    )


class InferResponse(BaseModel):
    frame_id: Optional[str] = None
    latency_ms: float
    detections: List[Detection]
    class_distribution: Dict[str, int]


class BatchInferResponse(BaseModel):
    total_images: int
    avg_latency_ms: float
    results: List[InferResponse]


class MetricsSnapshot(BaseModel):
    avg_latency_ms: float
    avg_fps: float
    total_frames: int
    class_distribution: Dict[str, int]
