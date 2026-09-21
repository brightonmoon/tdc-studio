"""Tests for Serving API and InferencePipeline."""

import pytest
from fastapi.testclient import TestClient

from tdc_studio.models.graph.graph_transformer import GraphTransformerModel
from tdc_studio.serving.app import app, set_pipeline
from tdc_studio.serving.pipeline import InferencePipeline


@pytest.fixture
def test_client():
    return TestClient(app)


def test_healthz_endpoint_initial(test_client):
    set_pipeline(None)
    resp = test_client.get("/healthz")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is False


def test_predict_endpoint_with_pipeline(test_client):
    # Initialize mock model & pipeline
    config = {"in_dim": 14, "hidden_dim": 16, "num_layers": 1, "task_type": "regression"}
    model = GraphTransformerModel(config)
    pipeline = InferencePipeline(model=model, is_dta=False)
    set_pipeline(pipeline)

    # 1. Healthcheck should show model_loaded = True
    resp_health = test_client.get("/healthz")
    assert resp_health.json()["model_loaded"] is True

    # 2. Predict on SMILES
    payload = {
        "smiles": ["CC(=O)O", "CCN"],
    }
    resp = test_client.post("/predict", json=payload)
    assert resp.status_code == 200
    res_data = resp.json()
    assert "predictions" in res_data
    assert len(res_data["predictions"]) == 2
    assert isinstance(res_data["predictions"][0], float)

    # Cleanup
    set_pipeline(None)


def test_predict_endpoint_not_ready(test_client):
    set_pipeline(None)
    payload = {"smiles": ["CCO"]}
    resp = test_client.post("/predict", json=payload)
    assert resp.status_code == 503
