from __future__ import annotations

import random
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np

from .schemas import Detection

TARGET_CLASSES = [
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


class BaseDetector:
    def predict(self, image: np.ndarray) -> List[Detection]:
        raise NotImplementedError


class DummyRoadDetector(BaseDetector):
    """Fallback detector for end-to-end integration demo without model dependency."""

    def predict(self, image: np.ndarray) -> List[Detection]:
        h, w = image.shape[:2]
        k = random.randint(1, 5)
        detections: List[Detection] = []
        for _ in range(k):
            x1 = random.randint(0, max(0, w - 30))
            y1 = random.randint(0, max(0, h - 30))
            x2 = min(w, x1 + random.randint(30, max(40, w // 4)))
            y2 = min(h, y1 + random.randint(30, max(40, h // 4)))
            detections.append(
                Detection(
                    label=random.choice(TARGET_CLASSES),
                    confidence=round(random.uniform(0.55, 0.98), 3),
                    bbox=[float(x1), float(y1), float(x2), float(y2)],
                )
            )
        return detections


class DetectorFactory:
    @staticmethod
    def create(model_path: str | None = None) -> BaseDetector:
        if model_path:
            path = Path(model_path)
            if path.exists():
                try:
                    from ultralytics import YOLO  # type: ignore

                    return UltralyticsDetector(YOLO(str(path)))
                except Exception:
                    pass
        return DummyRoadDetector()


class UltralyticsDetector(BaseDetector):
    def __init__(self, model) -> None:
        self.model = model

    def predict(self, image: np.ndarray) -> List[Detection]:
        result = self.model.predict(image, verbose=False)[0]
        names = result.names
        detections: List[Detection] = []

        for box in result.boxes:
            cls_id = int(box.cls.item())
            label = str(names[cls_id])
            if label not in TARGET_CLASSES:
                continue
            xyxy = box.xyxy.tolist()[0]
            detections.append(
                Detection(
                    label=label,
                    confidence=round(float(box.conf.item()), 3),
                    bbox=[float(v) for v in xyxy],
                )
            )
        return detections


def distribution(detections: List[Detection]) -> Dict[str, int]:
    return dict(Counter(d.label for d in detections))


def timed_predict(detector: BaseDetector, image: np.ndarray) -> tuple[List[Detection], float]:
    start = time.perf_counter()
    dets = detector.predict(image)
    latency_ms = (time.perf_counter() - start) * 1000
    return dets, round(latency_ms, 2)
