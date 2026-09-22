"""FastAPI serving application with lifespan management and non-blocking inference."""

import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from starlette.concurrency import run_in_threadpool

from tdc_studio.serving.pipeline import InferencePipeline
from tdc_studio.serving.schema import HealthResponse, InferenceRequest, InferenceResponse

logger = logging.getLogger("tdc_studio.serving")

# Global pipeline instance and metadata
_pipeline: Optional[InferencePipeline] = None
_model_meta: dict = {}


def set_pipeline(pipeline: Optional[InferencePipeline], meta: Optional[dict] = None) -> None:
    """Setter for global inference pipeline (used in startup or test injection)."""
    global _pipeline, _model_meta
    _pipeline = pipeline
    _model_meta = meta or {}


def get_pipeline() -> Optional[InferencePipeline]:
    """Getter for global inference pipeline."""
    return _pipeline


def get_model_meta() -> dict:
    """Getter for loaded model metadata."""
    return _model_meta


def init_pipeline_from_directory(model_dir: str) -> Optional[InferencePipeline]:
    """Load model from directory containing model.pt / best_model.pt and config.json."""
    if not os.path.isdir(model_dir):
        return None

    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        return None

    try:
        from tdc_studio.serving.exporter import load_model_from_checkpoint

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        model = load_model_from_checkpoint(model_dir)
        is_dta = config.get("type", "").endswith("_dta") or config.get("is_dta", False)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        pipeline = InferencePipeline(model=model, device=device, is_dta=is_dta)
        set_pipeline(pipeline, meta=config)
        logger.info("Successfully loaded model from '%s' on %s.", model_dir, device)
        return pipeline
    except Exception as e:
        logger.warning("Failed to load model from '%s': %s", model_dir, e)
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    # 1. Attempt loading model from environment variable or standard default paths
    model_dir = os.environ.get("MODEL_DIR")
    if not model_dir:
        for candidate in ["models/export", "models/checkpoint"]:
            if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, "config.json")):
                model_dir = candidate
                break

    if model_dir:
        init_pipeline_from_directory(model_dir)

    yield

    # Cleanup on shutdown
    set_pipeline(None)


app = FastAPI(
    title="TDC-Studio Inference Service",
    version="0.1.0",
    description="High-performance molecular property prediction microservice.",
    lifespan=lifespan,
)


@app.get("/healthz", response_model=HealthResponse)
def health_check():
    """Liveness / Readiness probe."""
    return HealthResponse(
        status="healthy",
        model_loaded=_pipeline is not None,
    )


@app.post("/predict", response_model=InferenceResponse)
async def predict(request: InferenceRequest):
    """Predict molecular properties or interactions."""
    pipeline = get_pipeline()
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model pipeline is not initialized or loaded yet. Start server with valid MODEL_DIR or load model checkpoint.",
        )

    try:
        # Offload CPU-heavy molecular featurization and PyTorch inference to threadpool
        preds = await run_in_threadpool(pipeline.predict, request.smiles, request.target_sequences)
        model_name = _model_meta.get("type", "TDC-Studio-Model")
        return InferenceResponse(
            predictions=preds,
            unit="score",
            model_name=model_name,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
