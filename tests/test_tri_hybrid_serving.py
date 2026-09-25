"""Tests for SOTA Tri-Hybrid Production Serving Pipeline and Exporter."""

import json
import os

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from api import app
from tdc_studio.models.graph.dmpnn import DMPNNModel
from tdc_studio.serving.app import set_pipeline
from tdc_studio.serving.exporter import export_tri_hybrid_package
from tdc_studio.serving.tri_hybrid_pipeline import (
    TriHybridInferencePipeline,
    load_tri_hybrid_from_package,
)


class DummyRegressor:
    """Mock GBDT or Ridge model for fast unit testing."""

    def __init__(self, constant_val: float = 2.0):
        self.val = constant_val

    def predict(self, X):
        return np.full(len(X), self.val, dtype=np.float32)


class DummyScaler:
    """Mock standard scaler."""

    def transform(self, X):
        return X


@pytest.fixture
def mock_tri_hybrid_components():
    cfg = {
        "type": "dmpnn",
        "in_dim": 14,
        "edge_dim": 6,
        "hidden_dim": 16,
        "num_layers": 1,
        "use_descriptors": True,
        "descriptor_dim": 210,
        "task_type": "regression",
    }
    dmpnn = DMPNNModel(cfg)
    gbdt = DummyRegressor(1.5)
    ridge = DummyRegressor(1.8)
    scaler = DummyScaler()
    weights = (0.2, 0.6, 0.2)
    calibration = (1.0, 0.0)
    stats = (0.0, 1.0)
    return dmpnn, gbdt, ridge, scaler, weights, calibration, stats, cfg


def test_tri_hybrid_inference_pipeline(mock_tri_hybrid_components):
    dmpnn, gbdt, ridge, scaler, weights, calib, stats, _ = mock_tri_hybrid_components
    pipeline = TriHybridInferencePipeline(
        dmpnn_model=dmpnn,
        gbdt_model=gbdt,
        ridge_model=ridge,
        scaler=scaler,
        blending_weights=weights,
        calibration_params=calib,
        dmpnn_stats=stats,
        device="cpu",
    )

    test_smiles = ["CC(=O)O", "CCN"]
    preds = pipeline.predict(test_smiles)

    assert len(preds) == 2
    for p in preds:
        assert isinstance(p, float)
        assert 0.0 <= p <= 100.0

    detailed = pipeline.predict_detailed(test_smiles)
    assert len(detailed) == 2
    assert "ppbr_percent" in detailed[0]
    assert "decision" in detailed[0]
    assert detailed[0]["unit"] == "%"


def test_export_and_load_tri_hybrid(tmp_path, mock_tri_hybrid_components):
    dmpnn, gbdt, ridge, scaler, weights, calib, stats, cfg = mock_tri_hybrid_components

    ckpt_path = str(tmp_path / "raw_dmpnn.pt")
    torch.save(dmpnn.state_dict(), ckpt_path)

    cfg_path = str(tmp_path / "raw_dmpnn_cfg.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    export_dir = str(tmp_path / "exported_sota")
    manifest = export_tri_hybrid_package(
        dmpnn_checkpoint_path=ckpt_path,
        dmpnn_config_path=cfg_path,
        gbdt_model=gbdt,
        ridge_model=ridge,
        scaler=scaler,
        blending_weights=weights,
        calibration_params=calib,
        dmpnn_stats=stats,
        output_dir=export_dir,
        benchmark_meta={"test_r2": 0.5412, "test_mae": 6.23},
    )

    assert manifest["model_type"] == "tri_hybrid_stacker"
    assert os.path.exists(os.path.join(export_dir, "ppbr_tri_hybrid_sota.pt"))
    assert os.path.exists(os.path.join(export_dir, "model.pt"))
    assert os.path.exists(os.path.join(export_dir, "config.json"))

    loaded_pipe = load_tri_hybrid_from_package(export_dir, device="cpu")
    preds = loaded_pipe.predict(["CCO"])
    assert len(preds) == 1
    assert 0.0 <= preds[0] <= 100.0


def test_api_tri_hybrid_serving(mock_tri_hybrid_components):
    dmpnn, gbdt, ridge, scaler, weights, calib, stats, _ = mock_tri_hybrid_components
    pipeline = TriHybridInferencePipeline(
        dmpnn_model=dmpnn,
        gbdt_model=gbdt,
        ridge_model=ridge,
        scaler=scaler,
        blending_weights=weights,
        calibration_params=calib,
        dmpnn_stats=stats,
        device="cpu",
    )

    set_pipeline(pipeline, meta={"type": "tri_hybrid_stacker", "task_type": "regression"})
    client = TestClient(app)

    # Health check
    health_resp = client.get("/healthz")
    assert health_resp.status_code == 200
    assert health_resp.json()["model_loaded"] is True

    # Predict
    resp = client.post("/predict", json={"smiles": ["CC(=O)O", "c1ccccc1"]})
    assert resp.status_code == 200
    data = resp.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 2
    assert data["model_name"] == "tri_hybrid_stacker"

    set_pipeline(None)
