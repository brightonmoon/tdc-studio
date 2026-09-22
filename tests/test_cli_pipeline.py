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
