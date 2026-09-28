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
    admet_model_loaded: bool = False
    vdss_model_loaded: bool = False
    dti_model_loaded: bool = False


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
