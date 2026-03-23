from __future__ import annotations

import json
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable


@dataclass
class Monitor:
    log_path: Path
    latency_window: deque = field(default_factory=lambda: deque(maxlen=200))
    fps_window: deque = field(default_factory=lambda: deque(maxlen=200))
    class_counter: Counter = field(default_factory=Counter)
    frame_count: int = 0

    def __post_init__(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, latency_ms: float, labels: Iterable[str]) -> None:
        now = time.time()
        self.latency_window.append(latency_ms)
        self.fps_window.append(now)
        label_list = list(labels)
        self.class_counter.update(label_list)
        self.frame_count += 1
        event = {
            "timestamp": now,
            "latency_ms": latency_ms,
            "labels": label_list,
            "frame_count": self.frame_count,
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def snapshot(self) -> Dict[str, object]:
        avg_latency = (
            sum(self.latency_window) / len(self.latency_window)
            if self.latency_window
            else 0.0
        )

        avg_fps = 0.0
        if len(self.fps_window) >= 2:
            elapsed = self.fps_window[-1] - self.fps_window[0]
            if elapsed > 0:
                avg_fps = (len(self.fps_window) - 1) / elapsed

        return {
            "avg_latency_ms": round(avg_latency, 2),
            "avg_fps": round(avg_fps, 2),
            "total_frames": self.frame_count,
            "class_distribution": dict(self.class_counter),
        }
