"""Comprehensive tests for ADMET Multi-Cluster DataModule and Phase 1 Smoke Tests."""

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from typer.testing import CliRunner

from tdc_studio.cli import app
from tdc_studio.core.registry import DATASETS
from tdc_studio.data.admet_cluster import ADMETClusterDataModule
from tdc_studio.models.graph.dmpnn import DMPNNModel
from tdc_studio.models.loss.multitask_loss import MaskedMultiTaskLoss


@pytest.fixture
def mock_distribution_df():
    """Synthetic dataframe for Cluster 2 (Distribution: PPBR, BBB, VDss, Lipo)."""
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
            "ppbr_az": [85.5, 32.0, 99.1, np.nan, 95.0, 80.0, 75.0, 20.0],
            "bbb_martins": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0],
            "vdss_lombardo": [0.15, 0.7, 0.1, 1.5, np.nan, 1.2, 0.2, 0.9],
            "lipophilicity_astrazeneca": [1.2, -0.1, 3.5, 2.8, np.nan, 3.1, 1.8, 0.5],
        }
    )


@pytest.fixture
def mock_cyp450_df():
    """Synthetic dataframe for Cluster 3 (CYP450 8-Head Matrix)."""
    smiles = [
        "CC(=O)OC1=CC=CC=C1C(=O)O",
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        "CCN(CC)C(=O)C1CN(C2CC3=CNC4=CC=CC(=C34)C2=C1)C",
        "CN(C)CCCN1C2=CC=CC=C2SC3=C1C=C(C=C3)Cl",
        "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",
    ]
    return pd.DataFrame(
        {
            "Drug": smiles,
            "cyp3a4_veith": [0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
            "cyp2d6_veith": [0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
            "cyp2c9_veith": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "cyp2c19_veith": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            "cyp1a2_veith": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            "cyp3a4_substrate_carbonmangels": [0.0, 0.0, 1.0, np.nan, 1.0, 0.0],
            "cyp2d6_substrate_carbonmangels": [0.0, 0.0, 0.0, 1.0, np.nan, 0.0],
            "cyp2c9_substrate_carbonmangels": [0.0, 0.0, 1.0, 0.0, 0.0, np.nan],
        }
    )


@pytest.fixture
def mock_herg_central_df():
    """Synthetic dataframe for hERG Central 3-head joint learning."""
    smiles = [
        "CC(=O)OC1=CC=CC=C1C(=O)O",
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        "CCN(CC)C(=O)C1CN(C2CC3=CNC4=CC=CC(=C34)C2=C1)C",
        "CN(C)CCCN1C2=CC=CC=C2SC3=C1C=C(C=C3)Cl",
        "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",
    ]
    return pd.DataFrame(
        {
            "Drug": smiles,
            "hERG_at_1uM": [2.5, 0.1, 15.2, 85.0, 92.1, 1.2],
            "hERG_at_10uM": [5.1, 1.0, 32.4, 98.0, 99.5, 4.0],
            "hERG_inhib": [0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
        }
    )


def test_admet_cluster_distribution_synthetic(mock_distribution_df):
    """Verify Cluster 2 mixed regression/classification with uncertainty."""
    tasks = [
        {"name": "ppbr_az", "category": "distribution", "type": "regression"},
        {"name": "bbb_martins", "category": "distribution", "type": "binary_classification"},
        {"name": "vdss_lombardo", "category": "distribution", "type": "regression"},
        {"name": "lipophilicity_astrazeneca", "category": "physicochemical", "type": "regression"},
    ]
    dm = ADMETClusterDataModule(
        tasks=tasks,
        synthetic_df=mock_distribution_df,
        primary_task="ppbr_az",
        standardize_target=True,
        use_descriptors=True,
    )
    dm.prepare_data()
    train_loader, val_loader, test_loader = dm.setup_loaders(batch_size=4)

    assert dm.num_tasks == 4
    assert dm.primary_task == "ppbr_az"
    assert "ppbr_az" in dm.task_stats
    assert "vdss_lombardo" in dm.task_stats

    batch = next(iter(train_loader))
    assert batch["labels"].shape == (4, 4)
    assert batch["mask"].shape == (4, 4)
    assert batch["descriptors"].shape == (4, 210)

    # Verify model forward with uncertainty loss
    model_cfg = {
        "type": "dmpnn_mtl",
        "in_dim": 14,
        "edge_dim": 6,
        "hidden_dim": 32,
        "num_layers": 2,
        "use_descriptors": True,
        "descriptor_dim": 210,
        "tasks": tasks,
        "use_uncertainty": True,
    }
    model = DMPNNModel(model_cfg)
    preds = model(batch)
    loss = model.compute_loss(preds, batch["labels"], mask=batch["mask"])
    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_admet_cluster_cyp450_synthetic(mock_cyp450_df):
    """Verify Cluster 3 8-Head multi-classification matrix with missing mask."""
    tasks = [
        {"name": "cyp3a4_veith", "type": "binary_classification"},
        {"name": "cyp2d6_veith", "type": "binary_classification"},
        {"name": "cyp2c9_veith", "type": "binary_classification"},
        {"name": "cyp2c19_veith", "type": "binary_classification"},
        {"name": "cyp1a2_veith", "type": "binary_classification"},
        {"name": "cyp3a4_substrate_carbonmangels", "type": "binary_classification"},
        {"name": "cyp2d6_substrate_carbonmangels", "type": "binary_classification"},
        {"name": "cyp2c9_substrate_carbonmangels", "type": "binary_classification"},
    ]
    dm = ADMETClusterDataModule(
        tasks=tasks,
        synthetic_df=mock_cyp450_df,
        primary_task="cyp3a4_veith",
        standardize_target=False,
    )
    dm.prepare_data()
    train_loader, _, _ = dm.setup_loaders(batch_size=4)

    assert dm.num_tasks == 8
    batch = next(iter(train_loader))
    assert batch["labels"].shape == (3, 8)
    assert batch["mask"].shape == (3, 8)

    model_cfg = {
        "type": "dmpnn_mtl",
        "in_dim": 14,
        "edge_dim": 6,
        "hidden_dim": 32,
        "num_layers": 2,
        "use_descriptors": True,
        "descriptor_dim": 210,
        "tasks": tasks,
    }
    model = DMPNNModel(model_cfg)
    preds = model(batch)
    loss = model.compute_loss(preds, batch["labels"], mask=batch["mask"])
    assert torch.isfinite(loss)


def test_admet_cluster_herg_central_synthetic(mock_herg_central_df):
    """Verify hERG Central 306k 3-Head mapping."""
    tasks = [
        {"name": "herg_central_at_1um", "label_name": "hERG_at_1uM", "type": "regression"},
        {"name": "herg_central_at_10um", "label_name": "hERG_at_10uM", "type": "regression"},
        {"name": "herg_central_inhib", "label_name": "hERG_inhib", "type": "binary_classification"},
    ]
    # Synthetic df already has the columns mapped
    mock_renamed = mock_herg_central_df.rename(
        columns={"hERG_at_1uM": "herg_central_at_1um", "hERG_at_10uM": "herg_central_at_10um", "hERG_inhib": "herg_central_inhib"}
    )
    dm = ADMETClusterDataModule(
        tasks=tasks,
        synthetic_df=mock_renamed,
        primary_task="herg_central_at_1um",
        standardize_target=True,
    )
    dm.prepare_data()
    train_loader, _, _ = dm.setup_loaders(batch_size=3)

    assert dm.num_tasks == 3
    batch = next(iter(train_loader))
    assert batch["labels"].shape == (3, 3)
    assert batch["mask"].shape == (3, 3)

    model_cfg = {
        "type": "dmpnn_mtl",
        "in_dim": 14,
        "edge_dim": 6,
        "hidden_dim": 32,
        "num_layers": 2,
        "use_descriptors": True,
        "descriptor_dim": 210,
        "tasks": tasks,
        "use_uncertainty": True,
    }
    model = DMPNNModel(model_cfg)
    preds = model(batch)
    loss = model.compute_loss(preds, batch["labels"], mask=batch["mask"])
    assert torch.isfinite(loss)


def test_cli_train_staged_herg_standalone_dry_run(tmp_path, mock_herg_central_df):
    """Verify CLI handles staged hierarchical configs (phase1_pretraining)."""
    class MockHergCentralDataModule(ADMETClusterDataModule):
        def __init__(self, **kwargs):
            mock_renamed = mock_herg_central_df.rename(
                columns={"hERG_at_1uM": "herg_central_at_1um", "hERG_at_10uM": "herg_central_at_10um", "hERG_inhib": "herg_central_inhib"}
            )
            kwargs["synthetic_df"] = mock_renamed
            super().__init__(**kwargs)

    DATASETS.register("mock_herg_central_loader")(MockHergCentralDataModule)

    config_content = {
        "phase1_pretraining": {
            "data": {
                "type": "mock_herg_central_loader",
                "dataset_name": "herg_central",
                "batch_size": 2,
                "primary_task": "herg_central_at_1um",
                "use_descriptors": True,
                "standardize_target": True,
                "tasks": [
                    {"name": "herg_central_at_1um", "type": "regression"},
                    {"name": "herg_central_at_10um", "type": "regression"},
                    {"name": "herg_central_inhib", "type": "binary_classification"},
                ],
            },
            "model": {
                "type": "dmpnn_mtl",
                "in_dim": 14,
                "edge_dim": 6,
                "hidden_dim": 16,
                "num_layers": 2,
                "use_descriptors": True,
                "descriptor_dim": 210,
                "tasks": [
                    {"name": "herg_central_at_1um", "type": "regression"},
                    {"name": "herg_central_at_10um", "type": "regression"},
                    {"name": "herg_central_inhib", "type": "binary_classification"},
                ],
            },
            "lr": 0.001,
            "max_epochs": 1,
            "eval_metric": "val_loss",
        },
        "tracking": {"enabled": False},
    }

    config_file = str(tmp_path / "herg_staged_config.yaml")
    with open(config_file, "w") as f:
        yaml.dump(config_content, f)

    chk_dir = str(tmp_path / "checkpoints_herg")
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["train", "--config", config_file, "--checkpoint-dir", chk_dir, "--dry-run"],
    )
    assert result.exit_code == 0, f"CLI train failed: {result.output}"
    assert "Starting Training Pipeline" in result.output
    assert "Training Pipeline Finished" in result.output
