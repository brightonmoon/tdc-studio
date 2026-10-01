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

from tdc_studio.explainability.attribution import MolecularExplainer
from tdc_studio.explainability.bioisostere import BioisostereRecommender
from tdc_studio.explainability.visualizer import AttributionVisualizer
from tdc_studio.serving.exporter import load_model_from_checkpoint

from tdc_studio.serving.pipeline import (
    DTIInferencePipeline,
    DTIMultiAffinityPipeline,
    InferencePipeline,
)
from tdc_studio.serving.schema import (
    BioisostereRecommendationItem,
    DTIInferenceRequest,
    DTIInferenceResponse,
    DTIMultiAffinityInferenceRequest,
    DTIMultiAffinityInferenceResponse,
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
    TherapeuticIndexRequest,
    TherapeuticIndexResponse,
    UnifiedADMETRequest,
    UnifiedADMETResponse,
)
from tdc_studio.serving.therapeutic_index_pipeline import (
    TherapeuticIndexPipeline,
    get_therapeutic_index_pipeline,
    set_therapeutic_index_pipeline,
)
from tdc_studio.serving.unified_pipeline import (
    UnifiedADMETPipeline,
    get_unified_pipeline,
    set_unified_pipeline,
)

logger = logging.getLogger("tdc_studio.serving")

# Global pipeline instances and metadata
_pipeline: Optional[InferencePipeline] = None
_vdss_pipeline: Optional[Any] = None
_pbpk_pipeline: Optional[Any] = None
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


def get_ti_pipeline() -> Optional[TherapeuticIndexPipeline]:
    """Getter for global TherapeuticIndexPipeline with on-demand fallback."""
    pipe = get_therapeutic_index_pipeline()
    if pipe is not None:
        return pipe
    # Auto-initialize from loaded DTI and UnifiedADMET pipelines if available
    admet_p = get_unified_pipeline()
    dti_p = get_dti_multi_pipeline() or get_dti_pipeline()
    if admet_p is not None or dti_p is not None:
        ti_pipe = TherapeuticIndexPipeline(dti_pipeline=dti_p, admet_pipeline=admet_p, device="cpu")
        set_therapeutic_index_pipeline(ti_pipe)
        return ti_pipe
    return None


def set_ti_pipeline(pipeline: Optional[TherapeuticIndexPipeline]) -> None:
    """Setter for global TherapeuticIndexPipeline."""
    set_therapeutic_index_pipeline(pipeline)


def init_pipeline_from_directory(model_dir: str) -> Optional[Any]:
    """Initialize InferencePipeline from an exported model directory."""
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        return None

    try:
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
    set_pbpk_pipeline(None)
    set_dti_pipeline(None)
    set_dti_multi_pipeline(None)
    set_unified_pipeline(None)
    set_ti_pipeline(None)


app = FastAPI(
    title="TDC-Studio Inference Service",
    version="0.2.0",
    description="High-performance molecular property, ADMET, and Drug-Target Interaction (DTI) prediction microservice.",
    lifespan=lifespan,
)


@app.get("/healthz", response_model=HealthResponse)
def health_check():
    """Liveness / Readiness probe."""
    has_admet = _pipeline is not None and not isinstance(_pipeline, DTIInferencePipeline)
    has_vdss = _vdss_pipeline is not None
    has_pbpk = _pbpk_pipeline is not None
    has_dti = _dti_pipeline is not None or isinstance(_pipeline, DTIInferencePipeline)
    unified_ready = get_unified_pipeline() is not None
    ti_ready = get_therapeutic_index_pipeline() is not None or unified_ready
    model_loaded = has_admet or has_dti or has_pbpk or unified_ready
    return HealthResponse(
        status="healthy",
        model_loaded=model_loaded,
        unified_ready=unified_ready,
        admet_model_loaded=has_admet,
        vdss_model_loaded=has_vdss,
        pbpk_pipeline_loaded=has_pbpk,
        dti_model_loaded=has_dti,
        therapeutic_index_ready=ti_ready,
    )


@app.post("/predict", response_model=InferenceResponse)
async def predict(request: InferenceRequest):
    """Predict molecular properties or generic interactions."""
    pipeline = get_pipeline()
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model pipeline is not initialized or exported yet. Start server with valid MODEL_DIR.",
        )

    try:
        preds = await run_in_threadpool(
            pipeline.predict,
            request.smiles,
            request.target_sequences,
        )
        return InferenceResponse(
            predictions=preds,
            unit=_model_meta.get("unit", "score"),
            model_name=_model_meta.get("type", "TDC-Studio-Model"),
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
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
_optimizer: Optional[Any] = None


def get_optimizer() -> Any:
    """Singleton getter for SelfCorrectingOptimizer."""
    global _optimizer
    if _optimizer is None:
        from tdc_studio.generative.lead_optimizer import SelfCorrectingOptimizer

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
    if optimizer.dti_pipeline is None:
        optimizer.dti_pipeline = get_dti_pipeline()

    try:
        report = await run_in_threadpool(
            optimizer.optimize,
            request.smiles,
            target_liability=request.target_liability,
            target_seq=request.target_sequence,
            weight_admet=request.weight_admet,
            weight_dta=request.weight_dta,
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
                parent_dta_pkd=c.parent_dta_pkd,
                candidate_dta_pkd=c.candidate_dta_pkd,
                dta_delta=c.dta_delta,
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
            target_protein_sequence=report.target_protein_sequence,
            parent_dta_pkd=report.parent_dta_pkd,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lead optimization error: {str(e)}")



# ------------------------------------------------------------------------------
# Drug-Target Interaction (DTI) Endpoints
# ------------------------------------------------------------------------------

@app.post("/predict/dti", response_model=DTIInferenceResponse)
async def predict_dti(request: DTIInferenceRequest):
    """Predict Drug-Target Interaction (DTI) binding affinities in pKd and Kd (nM)."""
    dti_pipe = get_dti_pipeline()
    if dti_pipe is None:
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
            conformal_lower_95=result.get("conformal_lower_95"),
            conformal_upper_95=result.get("conformal_upper_95"),
            confidence_interval_width=result.get("confidence_interval_width"),
            is_in_domain=result.get("is_in_domain"),
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


@app.post("/predict/therapeutic-index", response_model=TherapeuticIndexResponse)
async def predict_therapeutic_index(request: TherapeuticIndexRequest):
    """Evaluate Drug-Target Interaction (DTI) linked Therapeutic Index and safety margins.

    Combines On-Target Efficacy (Kd) with hERG Cardiotoxicity (IC50) and DILI Hepatotoxicity
    to compute logarithmic safety margin (TI), penalty adjustments, and overall clinical progression score.
    """
    ti_pipe = get_ti_pipeline()
    if ti_pipe is None:
        dti_p = get_dti_multi_pipeline() or get_dti_pipeline()
        admet_p = get_unified_pipeline()
        ti_pipe = TherapeuticIndexPipeline(dti_pipeline=dti_p, admet_pipeline=admet_p)
        set_therapeutic_index_pipeline(ti_pipe)

    t0 = time.perf_counter()
    try:
        items = await run_in_threadpool(
            ti_pipe.evaluate_batch,
            request.smiles,
            request.target_sequences,
            request.herg_source,
            request.herg_cutoff_nm,
            request.include_admet_details,
        )
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return TherapeuticIndexResponse(
            results=items,
            count=len(items),
            elapsed_ms=elapsed_ms,
            pipeline_version="TDC-Studio-TI-v1",
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Therapeutic Index evaluation error: {str(e)}")

