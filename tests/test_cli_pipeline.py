"""Integration tests for TDC-Studio CLI commands."""

import os

import yaml
from typer.testing import CliRunner

from tdc_studio.cli import app
from tdc_studio.models.graph.graph_transformer import GraphTransformerModel
from tdc_studio.serving.exporter import export_model_checkpoint

runner = CliRunner()


def test_cli_export_command(tmp_path):
    checkpoint_dir = str(tmp_path / "checkpoint")
    output_dir = str(tmp_path / "export")

    # Create dummy checkpoint
    config = {
        "type": "graph_transformer",
        "in_dim": 14,
        "hidden_dim": 16,
        "num_layers": 1,
        "task_type": "regression",
    }
    model = GraphTransformerModel(config)
    export_model_checkpoint(
        model=model,
        model_config=config,
        output_dir=checkpoint_dir,
        checkpoint_name="best_model.pt",
        extra_meta={"val_mae": 0.15},
    )

    result = runner.invoke(
        app,
        ["export", "--checkpoint-dir", checkpoint_dir, "--output-dir", output_dir],
    )
    assert result.exit_code == 0
    assert "Model successfully exported" in result.output
    assert os.path.exists(os.path.join(output_dir, "model.pt"))
    assert os.path.exists(os.path.join(output_dir, "export_manifest.json"))


def test_cli_train_dry_run_with_synthetic_data(tmp_path, dummy_smiles_df, monkeypatch):
    from tdc_studio.data.single_pred import ADMETDataModule

    class SyntheticADMETDataModule(ADMETDataModule):
        def __init__(self, **kwargs):
            super().__init__(dataset_name="synthetic", synthetic_df=dummy_smiles_df, **kwargs)

    from tdc_studio.core.registry import DATASETS

    DATASETS.register("synthetic_loader")(SyntheticADMETDataModule)

    config_content = {
        "data": {
            "type": "synthetic_loader",
            "dataset_name": "synthetic",
            "batch_size": 2,
            "metric_name": "mae",
            "params": {},
        },
        "model": {
            "type": "graph_transformer",
            "in_dim": 14,
            "hidden_dim": 16,
            "num_layers": 1,
        },
        "lr": 0.001,
        "max_epochs": 1,
        "tracking": {"enabled": False},
    }

    config_file = str(tmp_path / "config.yaml")
    with open(config_file, "w") as f:
        yaml.dump(config_content, f)

    chk_dir = str(tmp_path / "checkpoints")
    result = runner.invoke(
        app,
        ["train", "--config", config_file, "--checkpoint-dir", chk_dir, "--dry-run"],
    )
    assert result.exit_code == 0
    assert "Starting Training Pipeline" in result.output
    assert "Training Pipeline Finished" in result.output


def test_cli_ensemble_dry_run_with_synthetic_data(tmp_path, dummy_smiles_df):
    from tdc_studio.core.registry import DATASETS
    from tdc_studio.data.single_pred import ADMETDataModule

    class DummyDataModule(ADMETDataModule):
        def __init__(self, **kwargs):
            super().__init__(dataset_name="dummy_ens", synthetic_df=dummy_smiles_df, **kwargs)

    DATASETS.register("dummy_ens_loader")(DummyDataModule)

    config_content = {
        "data": {
            "type": "dummy_ens_loader",
            "dataset_name": "dummy_ens",
            "batch_size": 2,
            "metric_name": "mae",
            "params": {},
        },
        "model": {
            "type": "dmpnn",
            "in_dim": 14,
            "edge_dim": 6,
            "hidden_dim": 16,
            "depth": 2,
            "use_descriptors": False,
        },
        "lr": 0.001,
        "max_epochs": 1,
        "tracking": {"enabled": False},
    }

    config_file = str(tmp_path / "ensemble_config.yaml")
    with open(config_file, "w") as f:
        yaml.dump(config_content, f)

    chk_dir = str(tmp_path / "ens_checkpoints")
    result = runner.invoke(
        app,
        [
            "ensemble",
            "--config",
            config_file,
            "--n-models",
            "2",
            "--seeds",
            "42,43",
            "--checkpoint-dir",
            chk_dir,
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert "Starting Ensemble Pipeline" in result.output
    assert "Ensemble Benchmark Results" in result.output
    assert os.path.exists(os.path.join(chk_dir, "ensemble_summary.json"))


def test_cli_dti_train_dry_run(tmp_path):
    """Test tdc-studio train CLI with DTI task config and dry-run execution."""
    config_content = {
        "task": "dti",
        "data": {
            "type": "dta_loader",
            "dataset_name": "BindingDB_Kd",
            "split_type": "cold_drug",
            "batch_size": 4,
            "max_samples": 20,
        },
        "model": {
            "type": "graph_dta",
            "drug_encoder": {
                "type": "gine",
                "in_dim": 14,
                "hidden_dim": 16,
                "num_layers": 1,
            },
            "target_encoder": {
                "type": "protein_cnn",
                "out_dim": 16,
                "embed_dim": 16,
                "num_filters": 16,
            },
            "fusion": {
                "type": "bilinear_fusion",
                "drug_dim": 32,
                "target_dim": 16,
                "hidden_dim": 32,
                "out_dim": 1,
            },
        },
        "training": {
            "batch_size": 4,
            "max_epochs": 1,
            "learning_rate": 0.001,
        },
        "tracking": {"enabled": False},
    }

    config_file = str(tmp_path / "dti_config.yaml")
    with open(config_file, "w") as f:
        yaml.dump(config_content, f)

    chk_dir = str(tmp_path / "dti_checkpoints")
    result = runner.invoke(
        app,
        ["train", "--local", "--config", config_file, "--checkpoint-dir", chk_dir, "--dry-run"],
    )
    assert result.exit_code == 0
    assert "Starting Training Pipeline" in result.output
    assert "Task: dta" in result.output
    assert "Training Pipeline Finished" in result.output

