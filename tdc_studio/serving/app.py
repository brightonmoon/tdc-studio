"""FastAPI serving application with lifespan management and non-blocking inference."""

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Optional

import torch
from fastapi import FastAPI, HTTPException
from starlette.concurrency import run_in_threadpool

from tdc_studio.serving.pipeline import (
    DTIInferencePipeline,
    DTIMultiAffinityPipeline,
    InferencePipeline,
)
from tdc_studio.serving.schema import (
    DTIInferenceRequest,
    DTIInferenceResponse,
    DTIMultiAffinityInferenceRequest,
    DTIMultiAffinityInferenceResponse,
    HealthResponse,
    InferenceRequest,
    InferenceResponse,
)

logger = logging.getLogger("tdc_studio.serving")

# Global pipeline instances and metadata
_pipeline: Optional[InferencePipeline] = None
_vdss_pipeline: Optional[Any] = None
_model_meta: dict = {}

_dti_pipeline: Optional[DTIInferencePipeline] = None
_dti_multi_pipeline: Optional[DTIMultiAffinityPipeline] = None
_dti_model_meta: dict = {}


def set_pipeline(pipeline: Optional[InferencePipeline], meta: Optional[dict] = None) -> None:
    """Setter for global inference pipeline (used in startup or test injection)."""
    global _pipeline, _model_meta
    _pipeline = pipeline
    _model_meta = meta or {}


def set_vdss_pipeline(pipeline: Optional[Any]) -> None:
    """Setter for global VDss inference pipeline."""
    global _vdss_pipeline
    _vdss_pipeline = pipeline


def get_pipeline() -> Optional[InferencePipeline]:
    """Getter for global inference pipeline."""
    return _pipeline


def get_vdss_pipeline() -> Optional[Any]:
    """Getter for global VDss inference pipeline."""
    return _vdss_pipeline


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


def set_dti_multi_pipeline(pipeline: Optional[DTIMultiAffinityPipeline], meta: Optional[dict] = None) -> None:
    """Setter for global DTI Multi-Affinity inference pipeline."""
    global _dti_multi_pipeline, _dti_model_meta
    _dti_multi_pipeline = pipeline
    if meta:
        _dti_model_meta = meta


def get_dti_multi_pipeline() -> Optional[DTIMultiAffinityPipeline]:
    """Getter for global DTI Multi-Affinity inference pipeline with automatic fallback."""
    if _dti_multi_pipeline is not None:
        return _dti_multi_pipeline
    if _dti_pipeline is not None:
        return DTIMultiAffinityPipeline(
            model=_dti_pipeline.model,
            device=_dti_pipeline.device,
            scaler_meta=_dti_pipeline.scaler_meta,
        )
    return None


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

        model_type = config.get("type", "")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if "tri_hybrid" in model_type or os.path.exists(
            os.path.join(model_dir, "ppbr_tri_hybrid_sota.pt")
        ):
            from tdc_studio.serving.tri_hybrid_pipeline import (
                load_tri_hybrid_from_package,
                load_vdss_tri_hybrid_from_package,
            )

            pipeline = load_tri_hybrid_from_package(model_dir, device=device)
            set_pipeline(pipeline, meta=config)
            logger.info(
                "Successfully loaded Tri-Hybrid SOTA model from '%s' on %s.", model_dir, device
            )

            vdss_pt = os.path.join(model_dir, "vdss_tri_hybrid_sota.pt")
            if os.path.exists(vdss_pt):
                vdss_pipe = load_vdss_tri_hybrid_from_package(
                    model_dir, device=device, ppbr_pipeline=pipeline
                )
                set_vdss_pipeline(vdss_pipe)
                logger.info(
                    "Successfully loaded VDss Tri-Hybrid SOTA model from '%s' on %s.",
                    model_dir,
                    device,
                )
            return pipeline

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
            or config.get("task_type") == "multi_dta"
        )

        if is_dta:
            is_multi = (
                config.get("out_dim") == 3
                or config.get("fusion", {}).get("out_dim") == 3
                or config.get("task_type") == "multi_dta"
            )
            if is_multi:
                pipeline = DTIMultiAffinityPipeline(
                    model=model, device=device, scaler_meta=scaler_meta
                )
                set_dti_multi_pipeline(pipeline, meta=config)
                set_dti_pipeline(pipeline, meta=config)
            else:
                pipeline = DTIInferencePipeline(
                    model=model, device=device, scaler_meta=scaler_meta
                )
                set_dti_pipeline(pipeline, meta=config)
                set_dti_multi_pipeline(
                    DTIMultiAffinityPipeline(model=model, device=device, scaler_meta=scaler_meta),
                    meta=config,
                )
            if _pipeline is None:
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


def load_all_serving_models() -> None:
    """Discover and load both ADMET and DTI models simultaneously."""
    admet_env = os.environ.get("ADMET_MODEL_DIR")
    dti_env = os.environ.get("DTI_MODEL_DIR")
    generic_env = os.environ.get("MODEL_DIR")

    if generic_env:
        init_pipeline_from_directory(generic_env)
    if admet_env:
        init_pipeline_from_directory(admet_env)
    if dti_env:
        init_pipeline_from_directory(dti_env)

    # Autodiscover ADMET pipeline if not yet initialized
    if get_pipeline() is None or isinstance(get_pipeline(), DTIInferencePipeline):
        for candidate in ["models/export", "models/checkpoint"]:
            cfg_p = os.path.join(candidate, "config.json")
            if os.path.isdir(candidate) and os.path.exists(cfg_p):
                try:
                    with open(cfg_p, "r", encoding="utf-8") as f:
                        cfg = json.load(f)
                    if not (cfg.get("type", "").endswith("_dta") or cfg.get("is_dta", False)):
                        init_pipeline_from_directory(candidate)
                        break
                except Exception:
                    pass

    # Autodiscover DTI pipeline if not yet initialized
    if get_dti_pipeline() is None:
        for candidate in ["models/dti/phase_c_adv", "models/dti/phase_c", "models/dti/phase_b", "models/export/dti"]:
            cfg_p = os.path.join(candidate, "config.json")
            if os.path.isdir(candidate) and os.path.exists(cfg_p):
                init_pipeline_from_directory(candidate)
                if get_dti_pipeline() is not None:
                    break


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    load_all_serving_models()

    yield

    # Cleanup on shutdown
    set_pipeline(None)
    set_vdss_pipeline(None)
    set_dti_pipeline(None)
    set_dti_multi_pipeline(None)


app = FastAPI(
    title="TDC-Studio Inference Service",
    version="0.2.0",
    description="High-performance molecular property & Drug-Target Affinity (DTA) prediction microservice.",
    lifespan=lifespan,
)


@app.get("/healthz", response_model=HealthResponse)
def health_check():
    """Liveness / Readiness probe."""
    has_admet = _pipeline is not None and not isinstance(_pipeline, DTIInferencePipeline)
    has_vdss = _vdss_pipeline is not None
    has_dti = _dti_pipeline is not None or isinstance(_pipeline, DTIInferencePipeline)
    return HealthResponse(
        status="healthy",
        model_loaded=has_admet or has_dti,
        admet_model_loaded=has_admet,
        vdss_model_loaded=has_vdss,
        dti_model_loaded=has_dti,
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


@app.post("/predict/vdss", response_model=InferenceResponse)
async def predict_vdss(request: InferenceRequest):
    """Predict Volume of Distribution (log10 L/kg) using SOTA Tri-Hybrid Stacker."""
    vdss_pipe = get_vdss_pipeline()
    if vdss_pipe is None:
        raise HTTPException(
            status_code=503,
            detail="VDss Tri-Hybrid pipeline is not initialized or exported yet.",
        )

    try:
        preds = await run_in_threadpool(vdss_pipe.predict, request.smiles)
        return InferenceResponse(
            predictions=preds,
            unit="log10(L/kg)",
            model_name="vdss_tri_hybrid_sota",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"VDss inference error: {str(e)}")


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
            request.return_attention,
            request.return_contact_map,
            request.top_k_residues,
            request.return_full_matrix,
        )
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        model_name = _dti_model_meta.get("type", _model_meta.get("type", "GraphDTA-PhaseC-CrossAttention"))

        return DTIInferenceResponse(
            predictions_pkd=result["predictions_pkd"],
            kd_nm=result.get("kd_nm"),
            attention_weights=result.get("attention_weights"),
            contact_maps=result.get("contact_maps"),
            top_contact_residues=result.get("top_contact_residues"),
            top_contact_atoms=result.get("top_contact_atoms"),
            pymol_commands=result.get("pymol_commands"),
            unit="pK_d (-log10 Kd)",
            model_name=model_name,
            count=len(request.smiles),
            elapsed_ms=elapsed_ms,
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DTI inference error: {str(e)}")


@app.post("/predict/dti/multi-affinity", response_model=DTIMultiAffinityInferenceResponse)
@app.post("/predict/dti/multi", response_model=DTIMultiAffinityInferenceResponse)
async def predict_dti_multi_affinity(request: DTIMultiAffinityInferenceRequest):
    """Predict Kd, Ki, and IC50 binding affinities simultaneously with Affinity Consistency Score (ACS)."""
    multi_pipe = get_dti_multi_pipeline()
    if multi_pipe is None:
        raise HTTPException(
            status_code=503,
            detail="DTI Multi-Affinity model pipeline is not initialized or loaded yet. Start server with valid DTI MODEL_DIR.",
        )

    t0 = time.perf_counter()
    try:
        result = await run_in_threadpool(
            multi_pipe.predict_multi_affinity,
            request.smiles,
            request.target_sequences,
            request.return_nm,
            request.return_contact_maps,
            request.top_k_residues,
            request.return_full_matrix,
        )
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        model_name = _dti_model_meta.get("type", "GraphDTA-MultiAffinity-SOTA")

        return DTIMultiAffinityInferenceResponse(
            predictions_pkd=result["predictions_pkd"],
            predictions_pki=result["predictions_pki"],
            predictions_pic50=result["predictions_pic50"],
            kd_nm=result.get("kd_nm"),
            ki_nm=result.get("ki_nm"),
            ic50_nm=result.get("ic50_nm"),
            consistency_scores=result.get("consistency_scores"),
            consistency_tiers=result.get("consistency_tiers"),
            contact_maps=result.get("contact_maps"),
            top_contact_residues=result.get("top_contact_residues"),
            top_contact_atoms=result.get("top_contact_atoms"),
            pymol_commands=result.get("pymol_commands"),
            model_name=model_name,
            count=len(request.smiles),
            elapsed_ms=elapsed_ms,
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DTI multi-affinity inference error: {str(e)}")

