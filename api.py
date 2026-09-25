"""Root FastAPI serving endpoint module for TDC-Studio.

Allows launching the inference microservice directly via:
    uvicorn api:app --reload
    python -m uvicorn api:app --port 8000
"""

from tdc_studio.serving.app import app

__all__ = ["app"]
