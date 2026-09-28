"""Pydantic schemas for inference requests and responses."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class InferenceRequest(BaseModel):
    """Payload for molecular property or interaction inference."""

    smiles: List[str] = Field(
        ..., description="List of drug SMILES strings to predict.", min_length=1
    )
    target_sequences: Optional[List[str]] = Field(
        None, description="Optional list of target amino acid sequences (for DTA models)."
    )


class InferenceResponse(BaseModel):
    """Inference response payload."""

    predictions: List[float] = Field(..., description="Model predictions.")
    unit: str = Field(default="score", description="Unit or property description.")
    model_name: str = Field(default="TDC-Studio-Model", description="Serving model identifier.")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    model_loaded: bool = False
    unified_ready: bool = False
    admet_model_loaded: bool = False
    vdss_model_loaded: bool = False
    pbpk_pipeline_loaded: bool = False
    dti_model_loaded: bool = False


class PBPKResponse(BaseModel):
    """PBPK prediction response payload."""

    results: List[dict] = Field(..., description="Calculated in vivo pharmacokinetic profiles.")
    model_name: str = Field(
        default="TDC-Studio-PBPK-Pipeline", description="Serving model identifier."
    )


# ------------------------------------------------------------------------------
# Unified 22 ADMET Full-Lifecycle & PBPK Schemas
# ------------------------------------------------------------------------------

class ADMETIndicatorResult(BaseModel):
    """Single ADMET property prediction and decision categorization."""

    name: str = Field(..., description="Benchmark task name.")
    category: str = Field(..., description="ADMET cluster (absorption, distribution, metabolism, excretion, toxicity).")
    value: Optional[float] = Field(None, description="Numerical prediction (for regression).")
    probability: Optional[float] = Field(None, description="Predicted probability (for classification).")
    unit: str = Field(default="", description="Measurement or physical unit.")
    decision: str = Field(..., description="Medicinal chemistry qualitative decision tier.")


class PBPKProfileResult(BaseModel):
    """Physiological Pharmacokinetic profile derived from PBPK simulation."""

    vdss_l_kg: float = Field(..., description="Steady-state volume of distribution (L/kg).")
    half_life_hours: float = Field(..., description="Elimination half-life t1/2 (hours).")
    fraction_unbound: float = Field(..., description="Plasma unbound fraction (fu).")
    cl_total_l_h_kg: float = Field(..., description="Total systemic clearance (L/h/kg).")
    hepatic_clearance_l_h_kg: float = Field(..., description="Well-stirred hepatic clearance CL_H (L/h/kg).")
    hepatic_extraction_ratio: float = Field(..., description="Hepatic extraction ratio (E_H).")
    max_oral_bioavailability: float = Field(..., description="Theoretical maximal oral bioavailability F_H (1 - E_H).")
    t_half_tier: str = Field(..., description="Clinical elimination half-life classification.")
    extraction_tier: str = Field(..., description="Hepatic extraction ratio classification (Low/Intermediate/High).")


class UnifiedADMETProfile(BaseModel):
    """Complete 22 ADMET indicators across C1-C5 and PBPK profile for a single molecule."""

    smiles: str = Field(..., description="Original input SMILES.")
    canonical_smiles: str = Field(..., description="Standardized canonical SMILES.")
    elapsed_ms: float = Field(..., description="End-to-end inference latency in milliseconds.")
    drug_likeness_score: float = Field(..., description="Overall drug-likeness composite score (0-100).")
    absorption: Dict[str, ADMETIndicatorResult] = Field(..., description="C1: Absorption & Permeability (6 tasks).")
    distribution: Dict[str, ADMETIndicatorResult] = Field(..., description="C2: Distribution & Penetration (3 tasks).")
    metabolism: Dict[str, ADMETIndicatorResult] = Field(..., description="C3: CYP450 8-Head Metabolism Matrix (8 tasks).")
    excretion: Dict[str, ADMETIndicatorResult] = Field(..., description="C4: Elimination & Clearance (3 tasks).")
    toxicity: Dict[str, ADMETIndicatorResult] = Field(..., description="C5: Cardiotoxicity & Safety Profile (2+ tasks).")
    pbpk: Optional[PBPKProfileResult] = Field(None, description="Integrated in vivo PBPK PK profile.")


class UnifiedADMETRequest(BaseModel):
    """Request payload for the Unified 22 ADMET + PBPK endpoint."""

    smiles: List[str] = Field(
        ..., description="List of drug SMILES strings to evaluate.", min_length=1
    )


class UnifiedADMETResponse(BaseModel):
    """Response payload for the Unified 22 ADMET + PBPK endpoint."""

    results: List[UnifiedADMETProfile] = Field(..., description="Full-lifecycle ADMET profiles per compound.")
    model_version: str = Field(default="TDC-Studio-Unified-v1", description="Serving model version identifier.")


class ExplainRequest(BaseModel):
    """Request payload for model explainability and bioisostere recommendations."""

    smiles: str = Field(..., description="Target molecule SMILES to explain.", min_length=1)
    liability_focus: Optional[str] = Field(
        None, description="Optional target liability focus ('herg', 'ames', 'dili', 'clearance')."
    )
    steps: int = Field(default=30, description="Integrated Gradients interpolation steps.")


class BioisostereRecommendationItem(BaseModel):
    """Bioisosteric modification suggestion."""

    transformation_name: str = Field(..., description="Name of the bioisostere rule.")
    liability_addressed: str = Field(..., description="Target pharmacological/toxicological liability.")
    original_smiles: str = Field(..., description="Original molecule SMILES.")
    modified_smiles: str = Field(..., description="Suggested analogue SMILES.")
    rationale: str = Field(..., description="Medicinal chemistry rationale.")


class ExplainResponse(BaseModel):
    """Response payload containing atom attributions, SVG heatmap, and bioisostere suggestions."""

    smiles: str = Field(..., description="Input molecule SMILES.")
    canonical_smiles: str = Field(..., description="Canonicalized SMILES.")
    predicted_score: float = Field(..., description="Model prediction score.")
    num_atoms: int = Field(..., description="Number of heavy atoms.")
    atom_attributions: List[float] = Field(..., description="Raw Integrated Gradients attributions.")
    normalized_attributions: List[float] = Field(..., description="Normalized attributions in [-1, 1].")
    hotspot_atoms: List[int] = Field(..., description="Atom indices identified as liability hotspots.")
    svg_data_uri: str = Field(..., description="Base64 Data URI of 2D highlighted molecular SVG.")
    bioisostere_recommendations: List[BioisostereRecommendationItem] = Field(
        default_factory=list, description="List of suggested bioisostere analogues."
    )


# ------------------------------------------------------------------------------
# Self-Correcting Generative Lead Optimization Schemas
# ------------------------------------------------------------------------------

class OptimizeRequest(BaseModel):
    """Request payload for closed-loop self-correcting lead optimization."""

    smiles: str = Field(..., description="Target lead molecule SMILES to optimize.", min_length=1)
    target_liability: Optional[str] = Field(
        None, description="Optional target liability key (e.g. 'herg', 'ames', 'dili', 'clearance')."
    )
    max_candidates: int = Field(default=5, description="Maximum number of top candidates to return.")
    sa_threshold: float = Field(default=4.0, description="Maximum synthetic accessibility score allowed.")


class LiabilityDiagnosticItem(BaseModel):
    """Diagnosed liability item."""

    liability_key: str = Field(..., description="Task or property key.")
    liability_name: str = Field(..., description="Full indicator name.")
    cluster: str = Field(..., description="Cluster name.")
    current_value: float = Field(..., description="Prediction value or probability.")
    threshold: float = Field(..., description="Safety threshold.")
    severity: float = Field(..., description="Violation severity.")
    category: str = Field(..., description="Liability category.")


class OptimizedCandidateItem(BaseModel):
    """Repaired molecular candidate."""

    smiles: str = Field(..., description="Optimized candidate SMILES.")
    transformation_name: str = Field(..., description="Bioisostere rule applied.")
    liability_addressed: str = Field(..., description="Target liability.")
    rationale: str = Field(..., description="Medicinal chemistry rationale.")
    parent_liability_value: float = Field(..., description="Parent liability value.")
    candidate_liability_value: float = Field(..., description="Candidate liability value.")
    liability_delta: float = Field(..., description="Improvement delta (positive is improved).")
    sa_score: float = Field(..., description="Synthetic accessibility score (1-10, <=3.5 is ideal).")
    scaffold_preserved: bool = Field(..., description="Whether Bemis-Murcko core scaffold is retained.")
    fitness_score: float = Field(..., description="Multi-objective Pareto fitness score.")


class OptimizeResponse(BaseModel):
    """Response payload from SelfCorrectingOptimizer."""

    input_smiles: str = Field(..., description="Original input SMILES.")
    canonical_smiles: str = Field(..., description="Canonicalized parent SMILES.")
    bemis_murcko_scaffold: str = Field(..., description="Extracted core scaffold SMILES.")
    primary_liability: Optional[LiabilityDiagnosticItem] = Field(
        None, description="Primary detected or targeted liability."
    )
    candidates_generated: int = Field(..., description="Total candidate variants generated.")
    candidates_passing_sa_filter: int = Field(..., description="Candidates passing SAScore filter.")
    top_candidates: List[OptimizedCandidateItem] = Field(
        default_factory=list, description="Top ranked optimized analogues."
    )


# ------------------------------------------------------------------------------
# Drug-Target Interaction (DTI) Schemas
# ------------------------------------------------------------------------------

class DTIInferenceRequest(BaseModel):
    """Payload for Drug-Target Interaction (DTI) affinity inference."""

    smiles: List[str] = Field(
        ..., description="List of drug SMILES strings to predict.", min_length=1
    )
    target_sequences: List[str] = Field(
        ..., description="List of target amino acid sequences.", min_length=1
    )
    return_kd_nm: bool = Field(
        default=True, description="Whether to include Kd values in nanomolar (nM) in response."
    )
    return_attention: bool = Field(
        default=False,
        description="Whether to include residue/token attention weight maps (for models supporting XAI).",
    )

    @model_validator(mode="after")
    def check_lengths_match(self) -> "DTIInferenceRequest":
        if len(self.smiles) != len(self.target_sequences):
            raise ValueError(
                f"Mismatch between number of SMILES ({len(self.smiles)}) "
                f"and target sequences ({len(self.target_sequences)}). They must be equal."
            )
        return self


class DTIInferenceResponse(BaseModel):
    """Response payload for DTI affinity inference."""

    predictions_pkd: List[float] = Field(..., description="Predicted pKd values (-log10 Kd).")
    kd_nm: Optional[List[float]] = Field(None, description="Predicted Kd values in nanomolar (nM).")
    attention_weights: Optional[List[Dict[str, Any]]] = Field(
        None, description="Attention weight maps per pair (e.g. attn_d2t, attn_t2d)."
    )
    unit: str = Field(
        default="pK_d (-log10 Kd)", description="Measurement unit of primary prediction."
    )
    model_name: str = Field(default="GraphDTA-Model", description="Serving model identifier.")
    count: int = Field(..., description="Number of drug-target pairs evaluated.")
    elapsed_ms: Optional[float] = Field(None, description="Inference latency in milliseconds.")


class DTIMultiAffinityInferenceRequest(BaseModel):
    """Payload for Multi-Affinity (Kd, Ki, IC50) inference."""

    smiles: List[str] = Field(
        ..., description="List of drug SMILES strings to predict.", min_length=1
    )
    target_sequences: List[str] = Field(
        ..., description="List of target amino acid sequences.", min_length=1
    )
    return_nm: bool = Field(
        default=True, description="Whether to include affinity values in nanomolar (nM)."
    )
    return_contact_maps: bool = Field(
        default=False, description="Whether to include 2D token-level contact maps (XAI)."
    )

    @model_validator(mode="after")
    def check_lengths_match(self) -> "DTIMultiAffinityInferenceRequest":
        if len(self.smiles) != len(self.target_sequences):
            raise ValueError(
                f"Mismatch between number of SMILES ({len(self.smiles)}) "
                f"and target sequences ({len(self.target_sequences)}). They must be equal."
            )
        return self


class DTIMultiAffinityInferenceResponse(BaseModel):
    """Response payload for Multi-Affinity (Kd, Ki, IC50) inference."""

    predictions_pkd: Optional[List[float]] = Field(None, description="Predicted pKd values.")
    predictions_pki: Optional[List[float]] = Field(None, description="Predicted pKi values.")
    predictions_pic50: Optional[List[float]] = Field(None, description="Predicted pIC50 values.")
    kd_nm: Optional[List[float]] = Field(None, description="Predicted Kd in nM.")
    ki_nm: Optional[List[float]] = Field(None, description="Predicted Ki in nM.")
    ic50_nm: Optional[List[float]] = Field(None, description="Predicted IC50 in nM.")
    contact_maps: Optional[List[Any]] = Field(
        None, description="2D Contact Maps [L_drug, L_target]."
    )
    model_name: str = Field(
        default="GraphDTA-MultiAffinity", description="Serving model identifier."
    )
    count: int = Field(..., description="Number of drug-target pairs evaluated.")
    elapsed_ms: Optional[float] = Field(None, description="Inference latency in milliseconds.")
