"""FastAPI serving application with lifespan management and non-blocking inference."""

import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Optional

import torch
from fastapi import FastAPI, HTTPException
from starlette.concurrency import run_in_threadpool

from tdc_studio.explainability.attribution import MolecularExplainer
from tdc_studio.explainability.bioisostere import BioisostereRecommender
from tdc_studio.explainability.visualizer import AttributionVisualizer
from tdc_studio.generative.lead_optimizer import SelfCorrectingOptimizer
from tdc_studio.serving.pipeline import InferencePipeline
from tdc_studio.serving.schema import (
    BioisostereRecommendationItem,
    ExplainRequest,
    ExplainResponse,
    HealthResponse,
    InferenceRequest,
    InferenceResponse,
    LiabilityDiagnosticItem,
    OptimizedCandidateItem,
    OptimizeRequest,
    OptimizeResponse,
    PBPKResponse,
    UnifiedADMETRequest,
    UnifiedADMETResponse,
)
from tdc_studio.serving.unified_pipeline import (
    UnifiedADMETPipeline,
    get_unified_pipeline,
    set_unified_pipeline,
)

logger = logging.getLogger("tdc_studio.serving")

# Global pipeline instance and metadata
_pipeline: Optional[InferencePipeline] = None
_vdss_pipeline: Optional[Any] = None
_pbpk_pipeline: Optional[Any] = None
_model_meta: dict = {}


def set_pipeline(pipeline: Optional[InferencePipeline], meta: Optional[dict] = None) -> None:
    """Setter for global inference pipeline (used in startup or test injection)."""
    global _pipeline, _model_meta
    _pipeline = pipeline
    _model_meta = meta or {}


def set_vdss_pipeline(pipeline: Optional[Any]) -> None:
    """Setter for global VDss inference pipeline."""
    global _vdss_pipeline
    _vdss_pipeline = pipeline


def set_pbpk_pipeline(pipeline: Optional[Any]) -> None:
    """Setter for global PBPK inference pipeline."""
    global _pbpk_pipeline
    _pbpk_pipeline = pipeline


def get_pipeline() -> Optional[InferencePipeline]:
    """Getter for global inference pipeline."""
    return _pipeline


def get_vdss_pipeline() -> Optional[Any]:
    """Getter for global VDss inference pipeline."""
    return _vdss_pipeline


def get_pbpk_pipeline() -> Optional[Any]:
    """Getter for global PBPK inference pipeline."""
    return _pbpk_pipeline


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
                from tdc_studio.serving.pbpk_pipeline import PBPKServingPipeline

                pbpk_pipe = PBPKServingPipeline(
                    ppbr_pipeline=pipeline,
                    vdss_pipeline=vdss_pipe,
                )
                set_pbpk_pipeline(pbpk_pipe)
                logger.info("Successfully initialized PBPKServingPipeline.")
            return pipeline

        model = load_model_from_checkpoint(model_dir)
        is_dta = config.get("type", "").endswith("_dta") or config.get("is_dta", False)

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
            if os.path.isdir(candidate):
                model_dir = candidate
                break

    if model_dir and os.path.isdir(model_dir):
        if os.path.exists(os.path.join(model_dir, "config.json")):
            init_pipeline_from_directory(model_dir)
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            unified_pipe = UnifiedADMETPipeline.from_exported_directory(model_dir, device=device)
            set_unified_pipeline(unified_pipe)
            logger.info("Successfully initialized UnifiedADMETPipeline.")
        except Exception as e:
            logger.warning("UnifiedADMETPipeline auto-init note: %s", e)

    yield

    # Cleanup on shutdown
    set_pipeline(None)
    set_unified_pipeline(None)


app = FastAPI(
    title="TDC-Studio Inference Service",
    version="0.2.0",
    description="High-performance molecular property prediction microservice with Unified 22 ADMET + PBPK engine.",
    lifespan=lifespan,
)


@app.get("/healthz", response_model=HealthResponse)
def health_check():
    """Liveness / Readiness probe."""
    unified_ready = get_unified_pipeline() is not None
    model_loaded = _pipeline is not None or unified_ready
    return HealthResponse(
        status="healthy",
        model_loaded=model_loaded,
        unified_ready=unified_ready,
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


@app.post("/predict/pbpk", response_model=PBPKResponse)
async def predict_pbpk(request: InferenceRequest):
    """Predict in vivo pharmacokinetic profile (CL_total, t1/2, Vdss, fu, extraction ratio) via PBPK."""
    pbpk_pipe = get_pbpk_pipeline()
    if pbpk_pipe is None:
        raise HTTPException(
            status_code=503,
            detail="PBPK pipeline is not initialized. Ensure PPBR and VDss models are loaded.",
        )

    try:
        results = await run_in_threadpool(pbpk_pipe.predict_pbpk, request.smiles)
        return PBPKResponse(
            results=results,
            model_name="TDC-Studio-PBPK-Pipeline",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PBPK inference error: {str(e)}")


@app.post("/predict/admet_full", response_model=UnifiedADMETResponse)
async def predict_admet_full(request: UnifiedADMETRequest):
    """Predict complete C1-C5 22 full-lifecycle ADMET indicators and PBPK simulation in a single call."""
    unified_pipe = get_unified_pipeline()
    if unified_pipe is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        unified_pipe = UnifiedADMETPipeline(device=device)
        set_unified_pipeline(unified_pipe)

    try:
        profiles = await run_in_threadpool(unified_pipe.predict_batch, request.smiles)
        return UnifiedADMETResponse(
            results=profiles,
            model_version="TDC-Studio-Unified-v1",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unified ADMET inference error: {str(e)}")


# ------------------------------------------------------------------------------
# Explainability & Bioisostere Recommendations
# ------------------------------------------------------------------------------
_explainer: Optional[MolecularExplainer] = None
_visualizer: Optional[AttributionVisualizer] = None
_bioisostere: Optional[BioisostereRecommender] = None


def get_explainer() -> MolecularExplainer:
    """Singleton getter for MolecularExplainer."""
    global _explainer
    if _explainer is None:
        pipe = get_pipeline()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if pipe is not None and hasattr(pipe, "model") and pipe.model is not None:
            model = pipe.model
        else:
            from tdc_studio.models.graph.dmpnn import DMPNNModel

            model = DMPNNModel(
                {
                    "type": "dmpnn",
                    "in_dim": 14,
                    "edge_dim": 6,
                    "hidden_dim": 64,
                    "num_layers": 2,
                    "use_descriptors": False,
                }
            )
        _explainer = MolecularExplainer(model=model, device=device)
    return _explainer


def get_visualizer() -> AttributionVisualizer:
    """Singleton getter for AttributionVisualizer."""
    global _visualizer
    if _visualizer is None:
        _visualizer = AttributionVisualizer()
    return _visualizer


def get_bioisostere() -> BioisostereRecommender:
    """Singleton getter for BioisostereRecommender."""
    global _bioisostere
    if _bioisostere is None:
        _bioisostere = BioisostereRecommender()
    return _bioisostere


@app.post("/explain", response_model=ExplainResponse)
async def explain_molecule(request: ExplainRequest):
    """Explain molecular liabilities using Integrated Gradients atom heatmaps and bioisostere suggestions."""
    explainer = get_explainer()
    visualizer = get_visualizer()
    recommender = get_bioisostere()

    try:
        attr_res = await run_in_threadpool(
            explainer.attribute,
            request.smiles,
        )
        svg_data_uri = visualizer.render_data_uri(
            attr_res["smiles"],
            attr_res["normalized_attributions"],
            legend=f"XAI Attribution (Score: {attr_res['predicted_score']:.2f})",
        )
        suggestions = recommender.recommend(
            attr_res["smiles"],
            liability_focus=request.liability_focus,
        )
        rec_items = [BioisostereRecommendationItem(**s) for s in suggestions]

        return ExplainResponse(
            smiles=request.smiles,
            canonical_smiles=attr_res["smiles"],
            predicted_score=attr_res["predicted_score"],
            num_atoms=attr_res["num_atoms"],
            atom_attributions=attr_res["atom_attributions"],
            normalized_attributions=attr_res["normalized_attributions"],
            hotspot_atoms=attr_res["hotspot_atoms"],
            svg_data_uri=svg_data_uri,
            bioisostere_recommendations=rec_items,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation error: {str(e)}")


# ------------------------------------------------------------------------------
# Closed-Loop Generative Lead Optimizer Endpoint
# ------------------------------------------------------------------------------
_optimizer: Optional[SelfCorrectingOptimizer] = None


def get_optimizer() -> SelfCorrectingOptimizer:
    """Singleton getter for SelfCorrectingOptimizer."""
    global _optimizer
    if _optimizer is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        unified_pipe = get_unified_pipeline()
        if unified_pipe is None:
            unified_pipe = UnifiedADMETPipeline(device=device)
            set_unified_pipeline(unified_pipe)
        explainer = get_explainer()
        recommender = get_bioisostere()
        _optimizer = SelfCorrectingOptimizer(
            pipeline=unified_pipe,
            explainer=explainer,
            recommender=recommender,
            device=device,
        )
    return _optimizer


@app.post("/optimize", response_model=OptimizeResponse)
async def optimize_molecule(request: OptimizeRequest):
    """Automatically diagnose, localize, and repair liabilities using closed-loop self-correction."""
    optimizer = get_optimizer()
    try:
        report = await run_in_threadpool(
            optimizer.optimize,
            request.smiles,
            target_liability=request.target_liability,
            max_candidates=request.max_candidates,
        )

        primary_item = None
        if report.primary_liability is not None:
            primary_item = LiabilityDiagnosticItem(
                liability_key=report.primary_liability.liability_key,
                liability_name=report.primary_liability.liability_name,
                cluster=report.primary_liability.cluster,
                current_value=report.primary_liability.current_value,
                threshold=report.primary_liability.threshold,
                severity=report.primary_liability.severity,
                category=report.primary_liability.category,
            )

        cand_items = [
            OptimizedCandidateItem(
                smiles=c.smiles,
                transformation_name=c.transformation_name,
                liability_addressed=c.liability_addressed,
                rationale=c.rationale,
                parent_liability_value=c.parent_liability_value,
                candidate_liability_value=c.candidate_liability_value,
                liability_delta=c.liability_delta,
                sa_score=c.sa_score,
                scaffold_preserved=c.scaffold_preserved,
                fitness_score=c.fitness_score,
            )
            for c in report.top_candidates
        ]

        return OptimizeResponse(
            input_smiles=report.input_smiles,
            canonical_smiles=report.canonical_smiles,
            bemis_murcko_scaffold=report.bemis_murcko_scaffold,
            primary_liability=primary_item,
            candidates_generated=report.candidates_generated,
            candidates_passing_sa_filter=report.candidates_passing_sa_filter,
            top_candidates=cand_items,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lead optimization error: {str(e)}")


