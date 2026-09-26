import os

import pytest
import torch
from fastapi.testclient import TestClient

from tdc_studio.models.dti.dta_model import GraphDTAModel
from tdc_studio.models.graph.graph_transformer import GraphTransformerModel
from tdc_studio.serving.app import (
    app,
    init_pipeline_from_directory,
    set_dti_pipeline,
    set_pipeline,
)
from tdc_studio.serving.exporter import (
    export_dti_package,
    export_model_checkpoint,
    export_production_package,
    load_model_from_checkpoint,
)
from tdc_studio.serving.pipeline import DTIInferencePipeline, InferencePipeline


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


def test_dti_predict_endpoint_not_ready(test_client):
    set_pipeline(None)
    set_dti_pipeline(None)
    payload = {
        "smiles": ["CC(=O)O"],
        "target_sequences": ["MSHHWGYGKHNGPEHWHKDFPIAKGERQ"],
    }
    resp = test_client.post("/predict/dti", json=payload)
    assert resp.status_code == 503


def test_dti_predict_endpoint_validation_error(test_client):
    # Length mismatch: 2 SMILES vs 1 target sequence
    payload = {
        "smiles": ["CC(=O)O", "CCN"],
        "target_sequences": ["MSHHWGYGKHNGPEHWHKDFPIAKGERQ"],
    }
    resp = test_client.post("/predict/dti", json=payload)
    assert resp.status_code == 422


def test_dti_predict_endpoint_with_pipeline(test_client):
    # Lightweight mock GraphDTAModel (Phase A configuration)
    model_cfg = {
        "type": "graph_dta",
        "drug_encoder": {"type": "gine", "in_dim": 14, "hidden_dim": 16, "num_layers": 1},
        "target_encoder": {"type": "protein_cnn", "out_dim": 16, "kernel_sizes": [3]},
        "fusion": {"hidden_dim": 32},
    }
    model = GraphDTAModel(model_cfg)
    scaler_meta = {
        "y_mean": 2.5,
        "y_std": 1.2,
        "log_transform": True,
    }
    pipeline = DTIInferencePipeline(model=model, scaler_meta=scaler_meta)
    set_dti_pipeline(pipeline, meta={"type": "graph_dta"})

    # 1. Healthcheck should show dti_model_loaded = True
    resp_health = test_client.get("/healthz")
    assert resp_health.status_code == 200
    assert resp_health.json()["dti_model_loaded"] is True

    # 2. Predict DTI
    payload = {
        "smiles": ["CC(=O)O", "CCN"],
        "target_sequences": [
            "MSHHWGYGKHNGPEHWHKDFPIAKGERQ",
            "MKTAYIAKQRQISFVKSHFSRQLEER",
        ],
        "return_kd_nm": True,
    }
    resp = test_client.post("/predict/dti", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "predictions_pkd" in data
    assert len(data["predictions_pkd"]) == 2
    assert "kd_nm" in data
    assert len(data["kd_nm"]) == 2
    assert data["count"] == 2
    assert data["elapsed_ms"] is not None
    assert isinstance(data["predictions_pkd"][0], float)
    assert isinstance(data["kd_nm"][0], float)

    # Cleanup
    set_dti_pipeline(None)


def test_dti_export_and_auto_loading(tmp_path, test_client):
    model_cfg = {
        "type": "graph_dta",
        "drug_encoder": {"type": "gine", "in_dim": 14, "hidden_dim": 16, "num_layers": 1},
        "target_encoder": {"type": "protein_cnn", "out_dim": 16, "kernel_sizes": [3]},
        "fusion": {"hidden_dim": 32},
    }
    model = GraphDTAModel(model_cfg)
    export_dir = str(tmp_path / "dti_export")
    scaler_meta = {"y_mean": 3.0, "y_std": 1.5, "log_transform": True}

    manifest = export_dti_package(
        model=model,
        model_config=model_cfg,
        output_dir=export_dir,
        scaler_meta=scaler_meta,
        extra_meta={"ci": 0.7464},
    )
    assert manifest["has_scaler"] is True
    assert os.path.exists(os.path.join(export_dir, "scaler.json"))
    assert os.path.exists(os.path.join(export_dir, "config.json"))
    assert os.path.exists(os.path.join(export_dir, "model.pt"))

    # Auto-initialize from directory
    pipe = init_pipeline_from_directory(export_dir)
    assert pipe is not None
    assert isinstance(pipe, DTIInferencePipeline)

    resp = test_client.get("/healthz")
    assert resp.json()["dti_model_loaded"] is True

    # Test /predict/dti
    payload = {
        "smiles": ["CCO"],
        "target_sequences": ["MSHHWGYGKHNGPEHWHKDFPIAKGERQ"],
    }
    resp_pred = test_client.post("/predict/dti", json=payload)
    assert resp_pred.status_code == 200
    res_data = resp_pred.json()
    assert len(res_data["predictions_pkd"]) == 1

    # Cleanup
    set_dti_pipeline(None)
    set_pipeline(None)


def test_load_model_from_wrapped_dict(tmp_path):
    model_cfg = {
        "type": "graph_transformer",
        "in_dim": 14,
        "hidden_dim": 16,
        "num_layers": 1,
        "task_type": "regression",
    }
    model = GraphTransformerModel(model_cfg)
    ckpt_dir = str(tmp_path / "wrapped_ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Save as training dict wrapper (Colab train script style)
    torch.save(
        {
            "epoch": 7,
            "model_state_dict": model.state_dict(),
            "val_ci": 0.7464,
            "config": model_cfg,
        },
        os.path.join(ckpt_dir, "best_model.pt"),
    )

    # Should load successfully without explicit config.json on disk
    loaded = load_model_from_checkpoint(ckpt_dir)
    assert isinstance(loaded, GraphTransformerModel)


def test_dti_predict_with_attention_weights(test_client):
    """Verify that return_attention=True returns attention_weights list in response."""
    model_cfg = {
        "type": "graph_dta",
        "drug_encoder": {"type": "gine", "in_dim": 14, "hidden_dim": 16, "num_layers": 1},
        "target_encoder": {"type": "protein_cnn", "out_dim": 16, "kernel_sizes": [3]},
        "fusion": {"type": "cross_attention", "hidden_dim": 16, "num_heads": 2},
    }
    model = GraphDTAModel(model_cfg)
    pipeline = DTIInferencePipeline(model=model, device="cpu")
    set_dti_pipeline(pipeline, meta=model_cfg)

    payload = {
        "smiles": ["CC(=O)OC1=CC=CC=C1C(=O)O"],
        "target_sequences": ["MSHHWGYGKHNGPEHWHKDFPIAKGERQ"],
        "return_attention": True,
    }
    resp = test_client.post("/predict/dti", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "attention_weights" in data
    assert data["attention_weights"] is not None
    assert len(data["attention_weights"]) == 1

    # Cleanup
    set_dti_pipeline(None)


def test_dti_pipeline_graph_skipping_optimization(monkeypatch):
    """Verify that ChemBERTa encoder skips graph_transform parsing overhead."""
    class DummyChemBERTa(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.out_dim = 256
        def extract_features(self, batch):
            b = len(batch["drug_smiles_str"])
            return torch.zeros((b, 256))

    model_cfg = {
        "type": "graph_dta",
        "drug_encoder": {"type": "chemberta_encoder"},
        "target_encoder": {"type": "protein_cnn", "out_dim": 16},
        "fusion": {"hidden_dim": 16},
    }
    model = GraphDTAModel(model_cfg)
    # inject dummy ChemBERTa
    model.drug_encoder = DummyChemBERTa()

    pipeline = DTIInferencePipeline(model=model, device="cpu")
    assert pipeline.is_graph_drug is False

    # Monkeypatch graph_transform to fail if called
    def boom(smiles):
        raise RuntimeError("graph_transform should not be called!")
    monkeypatch.setattr(pipeline, "graph_transform", boom)

    preds = pipeline.predict(["CCO"], ["MSHHWGYGKHNGPEHWHKDFPIAKGERQ"])
    assert len(preds) == 1


def test_concurrent_admet_and_dti_serving(test_client):
    """Verify that both ADMET and DTI pipelines can run concurrently on single server."""
    # 1. Setup ADMET property model
    prop_cfg = {"type": "graph_transformer", "in_dim": 14, "hidden_dim": 16, "num_layers": 1}
    prop_model = GraphTransformerModel(prop_cfg)
    prop_pipe = InferencePipeline(model=prop_model, device="cpu")
    set_pipeline(prop_pipe, meta=prop_cfg)

    # 2. Setup DTI model
    dti_cfg = {
        "type": "graph_dta",
        "drug_encoder": {"type": "gine", "in_dim": 14, "hidden_dim": 16, "num_layers": 1},
        "target_encoder": {"type": "protein_cnn", "out_dim": 16, "kernel_sizes": [3]},
        "fusion": {"hidden_dim": 16},
    }
    dti_model = GraphDTAModel(dti_cfg)
    dti_pipe = DTIInferencePipeline(model=dti_model, device="cpu")
    set_dti_pipeline(dti_pipe, meta=dti_cfg)

    # Health check should report both
    resp = test_client.get("/healthz")
    assert resp.status_code == 200
    h_data = resp.json()
    assert h_data["admet_model_loaded"] is True
    assert h_data["dti_model_loaded"] is True
    assert h_data["model_loaded"] is True

    # /predict should hit ADMET
    resp_admet = test_client.post("/predict", json={"smiles": ["CC"]})
    assert resp_admet.status_code == 200

    # /predict/dti should hit DTI
    resp_dti = test_client.post(
        "/predict/dti",
        json={"smiles": ["CC"], "target_sequences": ["MSHHWGYGKHNGPEHWHKDFPIAKGERQ"]},
    )
    assert resp_dti.status_code == 200

    # Cleanup
    set_pipeline(None)
    set_dti_pipeline(None)


