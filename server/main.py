"""
YOLO Vision Dashboard - FastAPI Backend
Main entry point
"""
import os
import sys
import logging
from contextlib import asynccontextmanager

# Add conda env Library/bin so onnxruntime can find cuDNN/CUDA DLLs
_conda_bin = os.path.join(os.path.dirname(sys.executable), 'Library', 'bin')
_conda_bin = os.path.normpath(_conda_bin)
if os.path.isdir(_conda_bin):
    os.add_dll_directory(_conda_bin)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)

from routes.auth import router as auth_router
from routes.records import router as records_router
from routes.users import router as users_router
from routes.inference import router as inference_router, load_model_on_startup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: load model on startup if MODEL_PATH is set."""
    model_path = os.getenv("MODEL_PATH", "")
    if model_path and os.path.exists(model_path):
        logger.info(f"Loading model from MODEL_PATH: {model_path}")
        load_model_on_startup(model_path)
    else:
        logger.info("No MODEL_PATH set. Model can be uploaded via /api/inference/upload-model")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="YOLO Vision Dashboard API",
    description="Backend API for YOLO object detection with CUDA acceleration",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS - allow frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://0.0.0.0:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routes
app.include_router(auth_router, prefix="/api/auth", tags=["Authentication"])
app.include_router(records_router, prefix="/api/records", tags=["Detection Records"])
app.include_router(users_router, prefix="/api/users", tags=["User Management"])
app.include_router(inference_router, prefix="/api/inference", tags=["YOLO Inference"])


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "message": "YOLO Vision Dashboard API is running"}


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "3001"))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run("main:app", host=host, port=port, reload=False)
