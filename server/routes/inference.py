"""
YOLO Inference routes with CUDA acceleration.
Handles model upload, image inference, and model status.
Uses onnxruntime-gpu with CUDAExecutionProvider.
"""
import os
import sys
import io
import base64
import logging
import tempfile
import time
from typing import List, Optional

# Must set PATH before onnxruntime is imported so CUDA DLLs are found
_site_packages = os.path.normpath(os.path.join(os.path.dirname(sys.executable), 'Lib', 'site-packages'))
_cuda_paths = [
    os.path.join(_site_packages, 'nvidia', 'cudnn', 'bin'),
    os.path.join(_site_packages, 'nvidia', 'cublas', 'bin'),
    os.path.join(_site_packages, 'nvidia', 'cuda_nvrtc', 'bin'),
]
for _p in _cuda_paths:
    if os.path.isdir(_p) and _p not in os.environ.get('PATH', ''):
        os.environ['PATH'] = _p + os.pathsep + os.environ['PATH']

import uuid
import cv2
import numpy as np
from PIL import Image
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from pydantic import BaseModel

from middleware import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter()

# ============================================
# Global model state
# ============================================
_model_session = None
_model_name = ""
_model_input_name = ""
_model_output_name = ""

# ============================================
# Video session buffer
# ============================================
# { session_id: { "frames": [(timestamp_ms, jpeg_bytes), ...], "width": int, "height": int } }
_video_sessions: dict = {}


# Custom classes matching frontend
CUSTOM_CLASSES = [
    "person", "rider", "car", "bus", "truck",
    "bike", "motorcycle", "traffic light", "traffic sign", "train"
]


# ============================================
# Pydantic models
# ============================================
class DetectionResult(BaseModel):
    classId: int
    className: str
    score: float
    box: List[float]


class InferenceRequest(BaseModel):
    image: str  # base64 encoded image
    conf_threshold: float = 0.45
    iou_threshold: float = 0.50


class InferenceResponse(BaseModel):
    success: bool
    detections: List[DetectionResult] = []
    inference_time_ms: float = 0
    message: str = ""


class ModelStatus(BaseModel):
    loaded: bool
    model_name: str
    provider: str


class VideoFrameRequest(BaseModel):
    session_id: str
    frame: str        # base64 JPEG with detections drawn
    timestamp_ms: int
    width: int
    height: int


class VideoClip(BaseModel):
    index: int
    start_sec: float
    end_sec: float
    data: str         # base64 MP4


class VideoFinalizeRequest(BaseModel):
    session_id: str
    fps: float = 1.0
    clip_duration_sec: int = 10


class VideoFinalizeResponse(BaseModel):
    success: bool
    clips: list
    total_frames: int


# ============================================
# Model loading
# ============================================
def load_model_on_startup(model_path: str):
    """Load ONNX model at startup (called from main.py lifespan)."""
    global _model_session, _model_name, _model_input_name, _model_output_name

    try:
        import onnxruntime as ort

        # Try CUDA first, fall back to CPU
        providers = []
        available = ort.get_available_providers()

        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
            logger.info("Using CUDA acceleration for inference")
        if "CPUExecutionProvider" in available:
            providers.append("CPUExecutionProvider")

        if not providers:
            providers = ["CPUExecutionProvider"]

        _model_session = ort.InferenceSession(model_path, providers=providers)
        _model_name = os.path.basename(model_path)
        _model_input_name = _model_session.get_inputs()[0].name
        _model_output_name = _model_session.get_outputs()[0].name

        active_provider = _model_session.get_providers()[0]
        logger.info(f"Model loaded: {_model_name}, Provider: {active_provider}")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def _get_active_provider() -> str:
    """Get the active execution provider name."""
    if _model_session:
        providers = _model_session.get_providers()
        if providers:
            # get_providers() returns providers in priority order;
            # the first one is what onnxruntime will use for execution.
            return providers[0]
        return "Unknown"
    return "None"


# ============================================
# Image preprocessing & postprocessing
# ============================================
def preprocess_image(image: Image.Image, model_size: int = 640) -> np.ndarray:
    """
    Preprocess PIL Image to NCHW float32 tensor [1, 3, H, W].
    Resize to model_size x model_size, normalize to [0, 1].
    """
    img = image.convert("RGB").resize((model_size, model_size), Image.BILINEAR)
    img_array = np.array(img, dtype=np.float32) / 255.0

    # HWC -> CHW -> NCHW
    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def postprocess(
    output: np.ndarray,
    img_width: int,
    img_height: int,
    model_size: int = 640,
    conf_threshold: float = 0.45,
    iou_threshold: float = 0.50,
) -> List[DetectionResult]:
    """
    Post-process YOLOv8/v11 output tensor.
    Expected shape: [1, num_classes+4, num_anchors] (e.g., [1, 14, 8400])
    """
    if output.ndim == 3:
        output = output[0]  # Remove batch dim -> [num_classes+4, num_anchors]

    num_classes = output.shape[0] - 4
    num_anchors = output.shape[1]

    detections = []

    for i in range(num_anchors):
        # Class scores are from index 4 onward
        class_scores = output[4:, i]
        max_score = float(np.max(class_scores))
        class_id = int(np.argmax(class_scores))

        if max_score > conf_threshold:
            # Extract box [cx, cy, w, h]
            cx = float(output[0, i])
            cy = float(output[1, i])
            w = float(output[2, i])
            h = float(output[3, i])

            # Convert to [x1, y1, x2, y2] and scale to original image size
            x1 = ((cx - w / 2) / model_size) * img_width
            y1 = ((cy - h / 2) / model_size) * img_height
            x2 = ((cx + w / 2) / model_size) * img_width
            y2 = ((cy + h / 2) / model_size) * img_height

            class_name = CUSTOM_CLASSES[class_id] if class_id < len(CUSTOM_CLASSES) else f"class_{class_id}"

            detections.append(DetectionResult(
                classId=class_id,
                className=class_name,
                score=max_score,
                box=[x1, y1, x2, y2],
            ))

    # Apply NMS
    detections = apply_nms(detections, iou_threshold)
    return detections


def apply_nms(detections: List[DetectionResult], iou_threshold: float) -> List[DetectionResult]:
    """Apply Non-Maximum Suppression."""
    detections.sort(key=lambda d: d.score, reverse=True)
    selected = []

    for det in detections:
        keep = True
        for sel in selected:
            if det.classId == sel.classId:
                iou = calculate_iou(det.box, sel.box)
                if iou > iou_threshold:
                    keep = False
                    break
        if keep:
            selected.append(det)

    return selected


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Calculate Intersection over Union."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0


# ============================================
# API Routes
# ============================================
@router.get("/status", response_model=ModelStatus)
async def model_status():
    """Check if a model is loaded and which provider is active."""
    return ModelStatus(
        loaded=_model_session is not None,
        model_name=_model_name or "未加载模型",
        provider=_get_active_provider(),
    )


@router.get("/models")
async def list_models():
    """List all saved ONNX model files on the server."""
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    if not os.path.exists(models_dir):
        return {"success": True, "models": [], "active_model": _model_name}

    models = []
    for f in os.listdir(models_dir):
        if f.endswith(".onnx") or f.endswith(".pt"):
            filepath = os.path.join(models_dir, f)
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            models.append({
                "name": f,
                "size_mb": round(size_mb, 2),
                "is_active": f == _model_name,
            })

    return {
        "success": True,
        "models": models,
        "active_model": _model_name,
    }


@router.post("/load/{model_name}")
async def load_saved_model(model_name: str):
    """Load a previously saved model by name."""
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    model_path = os.path.join(models_dir, model_name)

    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"模型文件不存在: {model_name}")

    try:
        load_model_on_startup(model_path)
        return {
            "success": True,
            "message": f"模型已切换: {model_name}",
            "model_name": _model_name,
            "provider": _get_active_provider(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"加载模型失败: {str(e)}")


@router.post("/upload-model")
async def upload_model(file: UploadFile = File(...)):
    """Upload an ONNX or PT model file and load it with CUDA acceleration."""
    global _model_session, _model_name, _model_input_name, _model_output_name

    if not file.filename or not (file.filename.endswith(".onnx") or file.filename.endswith(".pt")):
        raise HTTPException(status_code=400, detail="请上传 .onnx 或 .pt 格式的模型文件")

    try:
        content = await file.read()
        models_dir = os.path.join(os.path.dirname(__file__), "models")
        os.makedirs(models_dir, exist_ok=True)
        save_path = os.path.join(models_dir, file.filename)

        with open(save_path, "wb") as f:
            f.write(content)

        # If PT file, convert to ONNX first
        if file.filename.endswith(".pt"):
            try:
                from ultralytics import YOLO as UltralyticsYOLO
            except ImportError:
                raise HTTPException(status_code=500, detail="PT 模型转换需要安装 ultralytics: pip install ultralytics")

            logger.info(f"Converting PT model to ONNX: {file.filename}")
            pt_model = UltralyticsYOLO(save_path)
            onnx_filename = file.filename.replace(".pt", ".onnx")
            onnx_path = os.path.join(models_dir, onnx_filename)
            pt_model.export(format="onnx", imgsz=640, opset=12, simplify=True, dynamic=False)
            # ultralytics exports next to the PT file
            exported_path = save_path.replace(".pt", ".onnx")
            if exported_path != onnx_path and os.path.exists(exported_path):
                import shutil
                shutil.move(exported_path, onnx_path)
            model_path = onnx_path
            logger.info(f"PT converted to ONNX: {onnx_filename}")
        else:
            model_path = save_path

        load_model_on_startup(model_path)

        return {
            "success": True,
            "message": f"模型加载成功: {_model_name}",
            "model_name": _model_name,
            "provider": _get_active_provider(),
            "input_names": [_model_input_name],
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"模型加载失败: {str(e)}")


@router.post("/detect", response_model=InferenceResponse)
async def detect(req: InferenceRequest):
    """
    Run YOLO inference on a base64-encoded image.
    Returns detected objects with bounding boxes and confidence scores.
    """
    if _model_session is None:
        raise HTTPException(status_code=400, detail="请先加载模型")

    try:
        # Decode base64 image
        # Handle data URL format: "data:image/jpeg;base64,..."
        image_data = req.image
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        img_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(img_bytes))
        img_width, img_height = img.size

        # Preprocess
        start_time = time.time()
        input_tensor = preprocess_image(img, model_size=640)

        # Run inference
        outputs = _model_session.run(
            [_model_output_name],
            {_model_input_name: input_tensor},
        )

        output = outputs[0]
        inference_time = (time.time() - start_time) * 1000  # ms

        # Postprocess
        detections = postprocess(
            output, img_width, img_height,
            model_size=640,
            conf_threshold=req.conf_threshold,
            iou_threshold=req.iou_threshold,
        )

        return InferenceResponse(
            success=True,
            detections=detections,
            inference_time_ms=round(inference_time, 2),
            message=f"检测到 {len(detections)} 个目标 ({inference_time:.1f}ms)",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"推理错误: {str(e)}")


@router.post("/video-frame")
async def submit_video_frame(req: VideoFrameRequest):
    """Buffer a single annotated frame for later video encoding."""
    global _video_sessions

    # Decode base64 JPEG
    image_data = req.frame
    if "," in image_data:
        image_data = image_data.split(",", 1)[1]
    jpeg_bytes = base64.b64decode(image_data)

    if req.session_id not in _video_sessions:
        _video_sessions[req.session_id] = {
            "frames": [],
            "width": req.width,
            "height": req.height,
        }

    _video_sessions[req.session_id]["frames"].append((req.timestamp_ms, jpeg_bytes))
    return {"success": True, "buffered": len(_video_sessions[req.session_id]["frames"])}


@router.post("/video-finalize", response_model=VideoFinalizeResponse)
async def finalize_video(req: VideoFinalizeRequest):
    """Encode buffered frames into MP4 clips and return them as base64."""
    global _video_sessions

    session = _video_sessions.pop(req.session_id, None)
    if not session or not session["frames"]:
        raise HTTPException(status_code=404, detail="Session not found or no frames buffered")

    frames = sorted(session["frames"], key=lambda x: x[0])  # sort by timestamp
    width = session["width"]
    height = session["height"]
    fps = max(req.fps, 0.5)
    frames_per_clip = max(1, int(fps * req.clip_duration_sec))

    import imageio

    clips = []
    for clip_idx, start in enumerate(range(0, len(frames), frames_per_clip)):
        chunk = frames[start:start + frames_per_clip]
        start_sec = chunk[0][0] / 1000.0
        end_sec = chunk[-1][0] / 1000.0

        tmp_path = os.path.join(tempfile.gettempdir(), f"yolo_{req.session_id}_{clip_idx}.mp4")

        with imageio.get_writer(
            tmp_path,
            fps=fps,
            codec="libx264",
            pixelformat="yuv420p",
            output_params=["-movflags", "+faststart"],
        ) as writer:
            for _, jpeg_bytes in chunk:
                img_array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                frame_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if frame_bgr is not None:
                    if frame_bgr.shape[1] != width or frame_bgr.shape[0] != height:
                        frame_bgr = cv2.resize(frame_bgr, (width, height))
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    writer.append_data(frame_rgb)

        with open(tmp_path, "rb") as f:
            mp4_bytes = f.read()
        os.remove(tmp_path)

        mp4_b64 = base64.b64encode(mp4_bytes).decode("utf-8")
        clips.append({
            "index": clip_idx,
            "start_sec": round(start_sec, 2),
            "end_sec": round(end_sec, 2),
            "data": mp4_b64,
        })
        logger.info(f"Encoded clip {clip_idx}: {len(chunk)} frames, {len(mp4_bytes)} bytes")

    return VideoFinalizeResponse(
        success=True,
        clips=clips,
        total_frames=len(frames),
    )
