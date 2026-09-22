"""Unit and integration tests for Phase 4 Bio-Permeability Multi-Task Learning pipeline."""

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from typer.testing import CliRunner

from tdc_studio.cli import app
from tdc_studio.core.registry import DATASETS
from tdc_studio.data.bio_permeability import BioPermeabilityDataModule
from tdc_studio.models.graph.dmpnn import DMPNNModel
from tdc_studio.models.loss.multitask_loss import MaskedMultiTaskLoss


@pytest.fixture
def mock_bio_permeability_df():
    """Synthetic dataset covering 4 bio-permeability tasks with intentional missing values."""
    smiles = [
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
        "CCN(CC)C(=O)C1CN(C2CC3=CNC4=CC=CC(=C34)C2=C1)C",  # LSD
        "CN(C)CCCN1C2=CC=CC=C2SC3=C1C=C(C=C3)Cl",  # Chlorpromazine
        "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",  # Testosterone
        "O=C(O)C1=CC=CC=C1O",  # Salicylic acid
        "CC(=O)NC1=CC=C(C=C1)O",  # Paracetamol
    ]
    return pd.DataFrame(
        {
            "Drug": smiles,
            "caco2_wang": [-4.5, -5.2, np.nan, -6.1, -4.9, -5.8, -4.3, np.nan],
            "lipophilicity_astrazeneca": [1.2, -0.1, 3.5, 2.8, np.nan, 3.1, 1.8, 0.5],
            "solubility_aqsoldb": [-1.5, -0.8, -2.5, -3.8, np.nan, -4.2, -1.1, -0.5],
            "hia_hou": [1.0, 1.0, 1.0, 0.0, np.nan, 1.0, 1.0, 1.0],
        }
    )


def test_bio_permeability_datamodule_synthetic(mock_bio_permeability_df):
    dm = BioPermeabilityDataModule(
        synthetic_df=mock_bio_permeability_df,
        modality="graph",
        use_descriptors=True,
        standardize_target=True,
        seed=42,
    )
    dm.prepare_data()
    train_loader, val_loader, test_loader = dm.setup_loaders(batch_size=4)

    assert dm.num_tasks == 4
    assert dm.primary_task == "caco2_wang"
    assert "caco2_wang" in dm.task_stats
    assert "lipophilicity_astrazeneca" in dm.task_stats
    assert "solubility_aqsoldb" in dm.task_stats

    # Check batch structure
    batch = next(iter(train_loader))
    assert "drug_graph" in batch
    assert "descriptors" in batch
    assert batch["descriptors"].shape == (4, 210)
    assert "labels" in batch
    assert "mask" in batch
    assert batch["labels"].shape == (4, 4)
    assert batch["mask"].shape == (4, 4)


def test_masked_multitask_loss_with_task_weights():
    task_names = ["caco2_wang", "lipophilicity_astrazeneca", "hia_hou"]
    task_types = ["regression", "regression", "binary_classification"]
    task_weights = {
        "caco2_wang": 2.0,
        "lipophilicity_astrazeneca": 0.5,
        "hia_hou": 0.5,
    }

    loss_fn = MaskedMultiTaskLoss(
        task_names=task_names,
        task_types=task_types,
        use_uncertainty=False,
        task_weights=task_weights,
    )

    preds = torch.tensor(
        [
            [-5.0, 1.5, 0.8],
            [-6.0, 2.0, -1.2],
        ],
        requires_grad=True,
    )
    targets = torch.tensor(
        [
            [-5.1, 1.0, 1.0],
            [-5.8, 2.2, 0.0],
        ]
    )
    mask = torch.tensor(
        [
            [True, True, True],
            [True, False, True],
        ]
    )

    total_loss, loss_dict = loss_fn(preds, targets, mask)
    assert total_loss.requires_grad
    assert total_loss.item() > 0.0
    assert "caco2_wang" in loss_dict

    total_loss.backward()
    assert preds.grad is not None


def test_dmpnn_multitask_model(mock_bio_permeability_df):
    tasks = [
        {"name": "caco2_wang", "type": "regression"},
        {"name": "lipophilicity_astrazeneca", "type": "regression"},
        {"name": "solubility_aqsoldb", "type": "regression"},
        {"name": "hia_hou", "type": "binary_classification"},
    ]
    model_cfg = {
        "in_dim": 14,
        "edge_dim": 6,
        "hidden_dim": 64,
        "num_layers": 2,
        "dropout": 0.1,
        "use_descriptors": True,
        "descriptor_dim": 210,
        "tasks": tasks,
        "task_weights": {
            "caco2_wang": 2.0,
            "lipophilicity_astrazeneca": 0.5,
            "solubility_aqsoldb": 0.5,
            "hia_hou": 0.5,
        },
    }

    model = DMPNNModel(model_cfg)
    assert hasattr(model, "task_heads")
    assert len(model.task_heads) == 4

    dm = BioPermeabilityDataModule(
        synthetic_df=mock_bio_permeability_df,
        modality="graph",
        use_descriptors=True,
        standardize_target=True,
        seed=42,
    )
    dm.prepare_data()
    loader, _, _ = dm.setup_loaders(batch_size=3)
    batch = next(iter(loader))

    preds = model(batch)
    assert preds.shape == (3, 4)

    loss = model.compute_loss(preds, batch["labels"], mask=batch["mask"])
    assert loss.item() > 0.0

    loss.backward()
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            assert torch.isfinite(p.grad).all()


def test_cli_train_bio_permeability_mtl_dry_run(tmp_path, mock_bio_permeability_df):
    class SyntheticBioPermDataModule(BioPermeabilityDataModule):
        def __init__(self, **kwargs):
            kwargs.setdefault("synthetic_df", mock_bio_permeability_df)
            kwargs.setdefault("use_descriptors", True)
            kwargs.setdefault("standardize_target", True)
            super().__init__(**kwargs)

    DATASETS.register("synthetic_bio_permeability_loader")(SyntheticBioPermDataModule)

    config_content = {
        "data": {
            "type": "synthetic_bio_permeability_loader",
            "dataset_name": "synthetic_bio_perm",
            "batch_size": 2,
            "primary_task": "caco2_wang",
            "metric_name": "r2",
            "use_descriptors": True,
            "standardize_target": True,
        },
        "model": {
            "type": "dmpnn_mtl",
            "in_dim": 14,
            "edge_dim": 6,
            "hidden_dim": 32,
            "num_layers": 2,
            "use_descriptors": True,
            "descriptor_dim": 210,
            "tasks": [
                {"name": "caco2_wang", "type": "regression"},
                {"name": "lipophilicity_astrazeneca", "type": "regression"},
                {"name": "solubility_aqsoldb", "type": "regression"},
                {"name": "hia_hou", "type": "binary_classification"},
            ],
            "task_weights": {"caco2_wang": 2.0, "lipophilicity_astrazeneca": 0.5},
        },
        "lr": 0.001,
        "max_epochs": 1,
        "eval_metric": "r2",
        "tracking": {"enabled": False},
    }

    config_file = str(tmp_path / "bio_mtl_config.yaml")
    with open(config_file, "w") as f:
        yaml.dump(config_content, f)

    chk_dir = str(tmp_path / "checkpoints")
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["train", "--config", config_file, "--checkpoint-dir", chk_dir, "--dry-run"],
    )
    assert result.exit_code == 0, f"CLI train failed: {result.output}"
    assert "Starting Training Pipeline" in result.output
    assert "Training Pipeline Finished" in result.output


def test_cli_ensemble_bio_permeability_mtl_dry_run(tmp_path, mock_bio_permeability_df):
    class SyntheticBioPermDataModule(BioPermeabilityDataModule):
        def __init__(self, **kwargs):
            kwargs.setdefault("synthetic_df", mock_bio_permeability_df)
            kwargs.setdefault("use_descriptors", True)
            kwargs.setdefault("standardize_target", True)
            super().__init__(**kwargs)

    DATASETS.register("synthetic_bio_permeability_loader2")(SyntheticBioPermDataModule)

    config_content = {
        "data": {
            "type": "synthetic_bio_permeability_loader2",
            "dataset_name": "synthetic_bio_perm",
            "batch_size": 2,
            "primary_task": "caco2_wang",
            "metric_name": "r2",
            "use_descriptors": True,
            "standardize_target": True,
        },
        "model": {
            "type": "dmpnn_mtl",
            "in_dim": 14,
            "edge_dim": 6,
            "hidden_dim": 32,
            "num_layers": 2,
            "use_descriptors": True,
            "descriptor_dim": 210,
            "tasks": [
                {"name": "caco2_wang", "type": "regression"},
                {"name": "lipophilicity_astrazeneca", "type": "regression"},
                {"name": "solubility_aqsoldb", "type": "regression"},
                {"name": "hia_hou", "type": "binary_classification"},
            ],
            "task_weights": {"caco2_wang": 2.0, "lipophilicity_astrazeneca": 0.5},
        },
        "lr": 0.001,
        "max_epochs": 1,
        "tracking": {"enabled": False},
    }

    config_file = str(tmp_path / "bio_mtl_ensemble_config.yaml")
    with open(config_file, "w") as f:
        yaml.dump(config_content, f)

    chk_dir = str(tmp_path / "checkpoints_ens")
    runner = CliRunner()
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
    assert result.exit_code == 0, f"CLI ensemble failed: {result.output}"
    assert "Starting Ensemble Pipeline" in result.output
    assert "Ensemble Benchmark Results" in result.output
