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
    conformal_uncertainty: Optional[Dict[str, Any]] = Field(
        None, description="Calibrated conformal prediction sets and intervals (coverage 1 - alpha)."
    )


class UnifiedADMETRequest(BaseModel):
    """Request payload for the Unified 22 ADMET + PBPK endpoint."""

    smiles: List[str] = Field(
        ..., description="List of drug SMILES strings to evaluate.", min_length=1
    )
    include_conformal: bool = Field(
        default=False, description="Whether to include conformal uncertainty quantification."
    )
    conformal_alpha: float = Field(
        default=0.10, ge=0.01, le=0.50, description="Significance level alpha for conformal coverage (default 0.10 for 90% confidence)."
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

class ResidueContactDetail(BaseModel):
    """Detailed structural contact information for a protein target residue."""

    rank: int = Field(..., description="Contact intensity rank (1-based).")
    index: int = Field(..., description="1-based sequence position (PDB/PyMOL compatible).")
    residue_name: str = Field(..., description="Single-letter amino acid code (e.g. 'E').")
    residue_code: str = Field(..., description="Standard 3-letter code with index (e.g. 'Glu312').")
    score: float = Field(..., description="Normalized contact intensity score (0.0 to 1.0).")


class AtomContactDetail(BaseModel):
    """Detailed structural contact information for a drug atom or token."""

    rank: int = Field(..., description="Interaction intensity rank.")
    atom_index: int = Field(..., description="Atom or subword token index.")
    token: str = Field(..., description="Atom symbol or ChemBERTa token string.")
    score: float = Field(..., description="Normalized interaction score (0.0 to 1.0).")


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
    return_contact_map: bool = Field(
        default=False,
        description="Whether to extract 2D contact maps, Top-K binding residues, and PyMOL commands.",
    )
    top_k_residues: int = Field(
        default=10, ge=1, le=50, description="Number of top contact residues to extract for PyMOL."
    )
    return_full_matrix: bool = Field(
        default=False, description="Whether to include raw full 2D float contact map matrix."
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
    contact_maps: Optional[List[Any]] = Field(
        None, description="2D Contact Maps [L_drug, L_target] if return_full_matrix=True."
    )
    top_contact_residues: Optional[List[List[ResidueContactDetail]]] = Field(
        None, description="Top-K contact residues per drug-target pair for structural pocket analysis."
    )
    top_contact_atoms: Optional[List[List[AtomContactDetail]]] = Field(
        None, description="Top-K interacting drug atoms or tokens per pair."
    )
    pymol_commands: Optional[List[str]] = Field(
        None, description="Ready-to-run PyMOL selection commands for 3D pocket visualization."
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
    top_k_residues: int = Field(
        default=10, ge=1, le=50, description="Number of top contact residues to extract for PyMOL."
    )
    return_full_matrix: bool = Field(
        default=False, description="Whether to include raw full 2D float contact map matrix."
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
    consistency_scores: Optional[List[float]] = Field(
        None, description="Affinity Consistency Score (ACS, 0-100) evaluating Kd/Ki/IC50 physical harmony."
    )
    consistency_tiers: Optional[List[str]] = Field(
        None, description="Qualitative consistency tier: 'High', 'Moderate', or 'Review Required'."
    )
    contact_maps: Optional[List[Any]] = Field(
        None, description="2D Contact Maps [L_drug, L_target]."
    )
    top_contact_residues: Optional[List[List[ResidueContactDetail]]] = Field(
        None, description="Top-K contact residues per drug-target pair."
    )
    top_contact_atoms: Optional[List[List[AtomContactDetail]]] = Field(
        None, description="Top-K interacting drug atoms or tokens."
    )
    pymol_commands: Optional[List[str]] = Field(
        None, description="Ready-to-run PyMOL selection commands for 3D pocket visualization."
    )
    model_name: str = Field(
        default="GraphDTA-MultiAffinity", description="Serving model identifier."
    )
    count: int = Field(..., description="Number of drug-target pairs evaluated.")
    elapsed_ms: Optional[float] = Field(None, description="Inference latency in milliseconds.")


# ------------------------------------------------------------------------------
# PBPK Virtual Population Monte Carlo Simulation Schemas
# ------------------------------------------------------------------------------

class PKMetricSummarySchema(BaseModel):
    mean: float
    sd: float
    cv_pct: float
    median: float
    p5: float
    p25: float
    p75: float
    p95: float


class ConcentrationTimeTrajectorySchema(BaseModel):
    time_hours: List[float]
    p5_ug_ml: List[float]
    median_ug_ml: List[float]
    p95_ug_ml: List[float]
    mean_ug_ml: List[float]


class VirtualPopulationRequest(BaseModel):
    smiles: str = Field(..., description="Molecular SMILES identifier.")
    subgroup: str = Field(
        default="healthy_adults",
        description="Target population: healthy_adults, renal_mild, renal_moderate, renal_severe, hepatic_child_pugh_a, hepatic_child_pugh_b, hepatic_child_pugh_c, geriatric",
    )
    n_subjects: int = Field(default=500, ge=10, le=5000, description="Virtual population subject count.")
    dose_mg: float = Field(default=100.0, gt=0, description="Single oral dose in mg.")
    ka_per_h: float = Field(default=1.2, gt=0, description="Oral absorption rate constant ka (1/h).")
    t_max_sim_hours: float = Field(default=48.0, gt=0, description="Concentration trajectory simulation window in hours.")


class VirtualPopulationMetricsSchema(BaseModel):
    vdss_l_kg: PKMetricSummarySchema
    cl_total_l_h_kg: PKMetricSummarySchema
    half_life_hours: PKMetricSummarySchema
    cmax_ug_ml: PKMetricSummarySchema
    tmax_hours: PKMetricSummarySchema
    auc_inf_ug_h_ml: PKMetricSummarySchema
    fraction_unbound: PKMetricSummarySchema


class VirtualPopulationResponse(BaseModel):
    subgroup: str
    n_subjects: int
    dose_mg: float
    metrics: VirtualPopulationMetricsSchema
    trajectory: ConcentrationTimeTrajectorySchema


# ------------------------------------------------------------------------------
# Retrosynthesis & Multi-Step Route Planning Schemas
# ------------------------------------------------------------------------------


class RetroCandidateItem(BaseModel):
    """Single candidate reactant set predicted for retrosynthetic disconnection."""

    reactants: str = Field(..., description="Precursor reactant SMILES separated by '.'")
    confidence: float = Field(..., description="Model confidence score or heuristic prior (0~1).")


class RetroSingleStepRequest(BaseModel):
    """Request payload for single-step retrosynthetic disconnection."""

    smiles: str = Field(..., description="Target product molecule SMILES.")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of precursor sets to generate.")
    reaction_type: Optional[int] = Field(
        None, ge=1, le=10, description="Optional USPTO-50K reaction class ID (1~10)."
    )


class RetroSingleStepResponse(BaseModel):
    """Response payload for single-step retrosynthesis."""

    product_smiles: str = Field(..., description="Input product SMILES.")
    candidates: List[RetroCandidateItem] = Field(..., description="Ranked precursor reactant candidates.")
    count: int = Field(..., description="Number of returned candidates.")


class ReactionStepSchema(BaseModel):
    """Individual reaction transformation step in a multi-step synthetic pathway."""

    step_number: int
    reactants: List[str]
    product: str
    rule_name: str
    confidence: float
    yield_pct: float
    cost: float


class RetroRouteItem(BaseModel):
    """Single complete retrosynthetic pathway with ranking and metrics."""

    rank: int = Field(..., description="Route rank (1 = champion, 2 = 1st alternative, etc.)")
    rank_score: float = Field(default=0.0, description="Multi-objective Pareto rank score.")
    target_smiles: str
    solved: bool
    total_depth: int
    cumulative_yield: float
    total_cost: float
    starting_materials: List[str]
    steps: List[ReactionStepSchema]
    mermaid_diagram: Optional[str] = None


class RouteComparisonItem(BaseModel):
    """Comparison matrix entry for a candidate route."""

    rank: int
    solved: bool
    total_depth: int
    cumulative_yield: float
    total_cost: float
    starting_materials_count: int
    starting_materials: List[str]
    reaction_rules: List[str]
    rank_score: float = 0.0


class RetroPlanRequest(BaseModel):
    """Request payload for multi-step retrosynthesis route planning."""

    smiles: str = Field(..., description="Target molecule SMILES to plan synthesis route for.")
    top_k: int = Field(
        default=3, ge=1, le=10, description="Number of candidate routes to return (1 = optimal only, >1 = include alternative routes)."
    )
    min_diversity: float = Field(
        default=0.25, ge=0.0, le=1.0, description="Minimum diversity distance between alternative routes."
    )
    banned_smiles: Optional[List[str]] = Field(
        default=None, description="Optional list of SMILES to ban/exclude from commercial stock."
    )
    max_depth: int = Field(default=5, ge=1, le=10, description="Maximum search tree depth.")
    timeout_sec: float = Field(default=5.0, ge=0.5, le=60.0, description="Search timeout in seconds.")
    render_mermaid: bool = Field(default=False, description="Whether to include rendered Mermaid diagram.")


class RetroPlanResponse(BaseModel):
    """Response payload containing complete retrosynthesis pathway(s)."""

    target_smiles: str
    solved: bool
    total_depth: int
    cumulative_yield: float
    total_cost: float
    starting_materials: List[str]
    steps: List[ReactionStepSchema]
    mermaid_diagram: Optional[str] = None
    routes: List[RetroRouteItem] = Field(
        default_factory=list, description="All top-k ranked routes (Rank 1 to K)."
    )
    comparison_summary: List[RouteComparisonItem] = Field(
        default_factory=list, description="Summary comparison table of all candidate routes."
    )


