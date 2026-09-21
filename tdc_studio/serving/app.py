"""FastAPI serving application with lifespan management and non-blocking inference."""

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from starlette.concurrency import run_in_threadpool

from tdc_studio.serving.pipeline import InferencePipeline
from tdc_studio.serving.schema import HealthResponse, InferenceRequest, InferenceResponse

# Global pipeline instance
_pipeline: Optional[InferencePipeline] = None


def set_pipeline(pipeline: Optional[InferencePipeline]) -> None:
    """Setter for global inference pipeline (used in startup or test injection)."""
    global _pipeline
    _pipeline = pipeline


def get_pipeline() -> Optional[InferencePipeline]:
    """Getter for global inference pipeline."""
    return _pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    # Pipeline initialization logic can load default model if available
    yield
    # Cleanup logic
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
            detail="Model pipeline is not initialized or loaded yet.",
        )

    try:
        # Offload CPU-heavy molecular featurization and PyTorch inference to threadpool
        preds = await run_in_threadpool(pipeline.predict, request.smiles, request.target_sequences)
        return InferenceResponse(
            predictions=preds,
            unit="score",
            model_name="TDC-Studio-Model",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
