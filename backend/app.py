from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import List

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .detector import DetectorFactory, distribution, timed_predict
from .monitor import Monitor
from .schemas import BatchInferResponse, InferResponse, MetricsSnapshot

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = os.getenv("MODEL_PATH")

app = FastAPI(title="Complex Road Multi-Object Detection Service", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=ROOT / "frontend"), name="static")

monitor = Monitor(log_path=ROOT / "logs" / "inference.log")
detector = DetectorFactory.create(MODEL_PATH)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "frontend" / "index.html")


async def decode_image(upload: UploadFile) -> np.ndarray:
    content = await upload.read()
    arr = np.frombuffer(content, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode image")
    return image


@app.post("/infer/image", response_model=InferResponse)
async def infer_image(file: UploadFile = File(...), frame_id: str | None = None) -> InferResponse:
    image = await decode_image(file)
    detections, latency_ms = timed_predict(detector, image)
    dist = distribution(detections)
    monitor.record(latency_ms, [d.label for d in detections])
    return InferResponse(
        frame_id=frame_id,
        latency_ms=latency_ms,
        detections=detections,
        class_distribution=dist,
    )


@app.post("/infer/batch", response_model=BatchInferResponse)
async def infer_batch(files: List[UploadFile] = File(...)) -> BatchInferResponse:
    results: List[InferResponse] = []
    latencies: List[float] = []

    for idx, f in enumerate(files):
        image = await decode_image(f)
        detections, latency_ms = timed_predict(detector, image)
        dist = distribution(detections)
        latencies.append(latency_ms)
        monitor.record(latency_ms, [d.label for d in detections])
        results.append(
            InferResponse(
                frame_id=str(idx),
                latency_ms=latency_ms,
                detections=detections,
                class_distribution=dist,
            )
        )

    avg = round(sum(latencies) / max(1, len(latencies)), 2)
    return BatchInferResponse(total_images=len(files), avg_latency_ms=avg, results=results)


@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        frame = base64.b64decode(data)
        arr = np.frombuffer(frame, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            await websocket.send_json({"error": "invalid frame"})
            continue

        detections, latency_ms = timed_predict(detector, image)
        dist = distribution(detections)
        monitor.record(latency_ms, [d.label for d in detections])
        await websocket.send_json(
            {
                "latency_ms": latency_ms,
                "detections": [d.model_dump() for d in detections],
                "class_distribution": dist,
                "metrics": monitor.snapshot(),
            }
        )


@app.get("/metrics", response_model=MetricsSnapshot)
def metrics() -> MetricsSnapshot:
    return MetricsSnapshot(**monitor.snapshot())
