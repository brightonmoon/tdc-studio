"""Pydantic schemas for inference requests and responses."""

from typing import List, Optional

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
    dti_model_loaded: bool = False



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
    unit: str = Field(default="pK_d (-log10 Kd)", description="Measurement unit of primary prediction.")
    model_name: str = Field(default="GraphDTA-Model", description="Serving model identifier.")
    count: int = Field(..., description="Number of drug-target pairs evaluated.")
    elapsed_ms: Optional[float] = Field(None, description="Inference latency in milliseconds.")

