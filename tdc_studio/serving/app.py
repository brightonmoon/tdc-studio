"""FastAPI serving application with lifespan management and non-blocking inference."""

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from starlette.concurrency import run_in_threadpool

from tdc_studio.serving.pipeline import DTIInferencePipeline, InferencePipeline
from tdc_studio.serving.schema import (
    DTIInferenceRequest,
    DTIInferenceResponse,
    HealthResponse,
    InferenceRequest,
    InferenceResponse,
)

logger = logging.getLogger("tdc_studio.serving")

# Global pipeline instances and metadata
_pipeline: Optional[InferencePipeline] = None
_model_meta: dict = {}

_dti_pipeline: Optional[DTIInferencePipeline] = None
_dti_model_meta: dict = {}


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


def set_dti_pipeline(pipeline: Optional[DTIInferencePipeline], meta: Optional[dict] = None) -> None:
    """Setter for global DTI inference pipeline."""
    global _dti_pipeline, _dti_model_meta
    _dti_pipeline = pipeline
    _dti_model_meta = meta or {}


def get_dti_pipeline() -> Optional[DTIInferencePipeline]:
    """Getter for global DTI inference pipeline."""
    return _dti_pipeline


def get_dti_model_meta() -> dict:
    """Getter for loaded DTI model metadata."""
    return _dti_model_meta


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

        scaler_path = os.path.join(model_dir, "scaler.json")
        scaler_meta = None
        if os.path.exists(scaler_path):
            with open(scaler_path, "r", encoding="utf-8") as f:
                scaler_meta = json.load(f)

        model = load_model_from_checkpoint(model_dir)
        is_dta = (
            config.get("type", "").endswith("_dta")
            or config.get("is_dta", False)
            or config.get("task_type") == "dta"
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if is_dta:
            pipeline = DTIInferencePipeline(
                model=model, device=device, scaler_meta=scaler_meta
            )
            set_dti_pipeline(pipeline, meta=config)
            # Also set global pipeline for backward compatibility with /predict
            set_pipeline(pipeline, meta=config)
            logger.info("Successfully loaded DTI pipeline from '%s' on %s.", model_dir, device)
        else:
            pipeline = InferencePipeline(model=model, device=device, is_dta=False)
            set_pipeline(pipeline, meta=config)
            logger.info("Successfully loaded property pipeline from '%s' on %s.", model_dir, device)

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
        for candidate in ["models/export", "models/checkpoint", "models/dti/phase_b"]:
            if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, "config.json")):
                model_dir = candidate
                break

    if model_dir:
        init_pipeline_from_directory(model_dir)

    yield

    # Cleanup on shutdown
    set_pipeline(None)
    set_dti_pipeline(None)


app = FastAPI(
    title="TDC-Studio Inference Service",
    version="0.2.0",
    description="High-performance molecular property & Drug-Target Affinity (DTA) prediction microservice.",
    lifespan=lifespan,
)


@app.get("/healthz", response_model=HealthResponse)
def health_check():
    """Liveness / Readiness probe."""
    return HealthResponse(
        status="healthy",
        model_loaded=_pipeline is not None or _dti_pipeline is not None,
        dti_model_loaded=_dti_pipeline is not None,
    )


@app.post("/predict", response_model=InferenceResponse)
async def predict(request: InferenceRequest):
    """Predict molecular properties or generic interactions."""
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


@app.post("/predict/dti", response_model=DTIInferenceResponse)
async def predict_dti(request: DTIInferenceRequest):
    """Predict Drug-Target Interaction (DTI) binding affinities in pKd and Kd (nM)."""
    dti_pipe = get_dti_pipeline()
    if dti_pipe is None:
        # Check if generic pipeline is a DTIInferencePipeline
        gen_pipe = get_pipeline()
        if isinstance(gen_pipe, DTIInferencePipeline):
            dti_pipe = gen_pipe

    if dti_pipe is None:
        raise HTTPException(
            status_code=503,
            detail="DTI model pipeline is not initialized or loaded yet. Start server with valid DTI MODEL_DIR.",
        )

    t0 = time.perf_counter()
    try:
        result = await run_in_threadpool(
            dti_pipe.predict_affinity,
            request.smiles,
            request.target_sequences,
            request.return_kd_nm,
        )
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        model_name = _dti_model_meta.get("type", _model_meta.get("type", "GraphDTA-PhaseB"))

        return DTIInferenceResponse(
            predictions_pkd=result["predictions_pkd"],
            kd_nm=result.get("kd_nm"),
            unit="pK_d (-log10 Kd)",
            model_name=model_name,
            count=len(request.smiles),
            elapsed_ms=elapsed_ms,
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DTI inference error: {str(e)}")

