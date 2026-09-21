"""Pydantic schemas for inference requests and responses."""

from typing import List, Optional

from pydantic import BaseModel, Field


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
