"""Tests for Serving API, Exporter, and InferencePipeline."""

import os

import pytest
from fastapi.testclient import TestClient

from tdc_studio.models.graph.graph_transformer import GraphTransformerModel
from tdc_studio.serving.app import app, init_pipeline_from_directory, set_pipeline
from tdc_studio.serving.exporter import (
    export_model_checkpoint,
    export_production_package,
    load_model_from_checkpoint,
)
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


def test_model_export_and_auto_loading(tmp_path, test_client):
    # 1. Train/Mock a model
    config = {
        "type": "graph_transformer",
        "in_dim": 14,
        "hidden_dim": 16,
        "num_layers": 1,
        "task_type": "regression",
    }
    model = GraphTransformerModel(config)
    checkpoint_dir = str(tmp_path / "checkpoint")

    # 2. Export checkpoint
    export_model_checkpoint(
        model=model,
        model_config=config,
        output_dir=checkpoint_dir,
        checkpoint_name="best_model.pt",
        extra_meta={"val_mae": 0.25},
    )
    assert os.path.exists(os.path.join(checkpoint_dir, "best_model.pt"))
    assert os.path.exists(os.path.join(checkpoint_dir, "config.json"))

    # 3. Test load_model_from_checkpoint with auto-class resolution (model_cls=None)
    loaded_model = load_model_from_checkpoint(checkpoint_dir)
    assert isinstance(loaded_model, GraphTransformerModel)

    # 4. Test export_production_package
    export_dir = str(tmp_path / "export")
    manifest = export_production_package(checkpoint_dir=checkpoint_dir, output_dir=export_dir)
    assert manifest["model_type"] == "graph_transformer"
    assert os.path.exists(os.path.join(export_dir, "model.pt"))
    assert os.path.exists(os.path.join(export_dir, "export_manifest.json"))

    # 5. Test auto-initialization from directory
    pipe = init_pipeline_from_directory(export_dir)
    assert pipe is not None

    resp = test_client.get("/healthz")
    assert resp.json()["model_loaded"] is True

    # Test prediction
    resp_pred = test_client.post("/predict", json={"smiles": ["CCO"]})
    assert resp_pred.status_code == 200
    assert len(resp_pred.json()["predictions"]) == 1

    # Cleanup
    set_pipeline(None)
