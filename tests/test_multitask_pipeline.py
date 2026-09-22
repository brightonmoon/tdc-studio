"""Tests for 3-Tier Categorical Multi-Task Learning and SMILES Modality Pipeline."""

import numpy as np
import pandas as pd
import pytest
import torch

from tdc_studio.data.multi_task import MultiTaskDataModule
from tdc_studio.data.transforms import (
    CanonicalSmilesNormalizer,
    RandomizedSmilesAugmenter,
)
from tdc_studio.models.hybrid.categorical_mtl import CategoricalMTLModel
from tdc_studio.models.loss.multitask_loss import MaskedMultiTaskLoss
from tdc_studio.serving.multitask_pipeline import (
    MultiTaskInferencePipeline,
    ThresholdDecisionEngine,
)


@pytest.fixture
def mock_multitask_df():
    """Synthetic multi-task ADMET dataset with intentional missing values."""
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
    # Mixture of regression and binary classification tasks with NaN values
    return pd.DataFrame(
        {
            "Drug": smiles,
            "caco2_wang": [-4.5, -5.2, np.nan, -6.1, -4.9, -5.8, -4.3, np.nan],
            "hia_hou": [1.0, 1.0, 1.0, 0.0, np.nan, 1.0, 1.0, 1.0],
            "herg": [0.0, 0.0, 0.0, np.nan, 1.0, 0.0, np.nan, 0.0],
            "dili": [0.0, 0.0, np.nan, 0.0, 1.0, np.nan, 0.0, 0.0],
            "solubility_aqsoldb": [-1.5, -0.8, -2.5, -3.8, np.nan, -4.2, -1.1, -0.5],
        }
    )


def test_smiles_normalizer_and_augmenter():
    normalizer = CanonicalSmilesNormalizer(remove_salts=True)
    # Salt mixture: Sodium acetate (CC(=O)[O-].[Na+]) -> acetic acid / acetate
    salt_smiles = "CC(=O)[O-].[Na+]"
    normalized = normalizer(salt_smiles)
    assert normalized is not None
    assert "." not in normalized  # Salt stripped, keeping largest organic fragment

    # Invalid SMILES handling
    assert normalizer("INVALID_XYZ_123") is None
    assert normalizer(None) is None

    # Augmenter
    augmenter = RandomizedSmilesAugmenter(p=1.0)
    aspirin = "CC(=O)OC1=CC=CC=C1C(=O)O"
    aug = augmenter(aspirin)
    assert isinstance(aug, str)
    assert len(aug) > 0


def test_multitask_data_module(mock_multitask_df):
    tasks = [
        {"name": "caco2_wang", "category": "absorption", "type": "regression"},
        {"name": "hia_hou", "category": "absorption", "type": "binary_classification"},
        {"name": "herg", "category": "toxicity", "type": "binary_classification"},
        {"name": "dili", "category": "toxicity", "type": "binary_classification"},
        {"name": "solubility_aqsoldb", "category": "physicochemical", "type": "regression"},
    ]

    dm = MultiTaskDataModule(
        tasks=tasks,
        synthetic_df=mock_multitask_df,
        modality="graph",
        seed=42,
    )
    dm.prepare_data()
    train_loader, val_loader, test_loader = dm.setup_loaders(batch_size=4)

    assert dm.num_tasks == 5
    assert len(dm.category_to_task_names["absorption"]) == 2
    assert len(dm.category_to_task_names["toxicity"]) == 2

    # Check batch structure
    batch = next(iter(train_loader))
    assert "drug_graph" in batch
    assert "labels" in batch
    assert "mask" in batch
    assert batch["labels"].shape == (4, 5)
    assert batch["mask"].shape == (4, 5)
    assert batch["mask"].dtype == torch.bool


def test_masked_multitask_loss():
    task_names = ["caco2", "herg", "solubility"]
    task_types = ["regression", "binary_classification", "regression"]

    loss_fn = MaskedMultiTaskLoss(task_names=task_names, task_types=task_types, use_uncertainty=True)

    # Batch of 3 samples, 3 tasks
    preds = torch.tensor(
        [
            [-5.0, 1.5, -2.0],
            [-6.0, -1.0, -1.5],
            [-4.5, 0.5, -3.0],
        ],
        requires_grad=True,
    )
    targets = torch.tensor(
        [
            [-5.1, 1.0, -2.1],
            [0.0, 0.0, -1.2],  # caco2 and herg missing for sample 2
            [-4.4, 1.0, 0.0],  # solubility missing for sample 3
        ]
    )
    mask = torch.tensor(
        [
            [True, True, True],
            [False, False, True],
            [True, True, False],
        ]
    )

    loss, loss_dict = loss_fn(preds, targets, mask)
    assert loss.requires_grad
    assert loss.item() > 0.0
    assert "caco2" in loss_dict
    assert "herg" in loss_dict

    # Backpropagation test (verifying gradients on uncertainty log_vars and predictions)
    loss.backward()
    assert preds.grad is not None
    assert loss_fn.log_vars.grad is not None


def test_categorical_mtl_model_architecture(mock_multitask_df):
    tasks = [
        {"name": "caco2_wang", "category": "absorption", "type": "regression"},
        {"name": "hia_hou", "category": "absorption", "type": "binary_classification"},
        {"name": "herg", "category": "toxicity", "type": "binary_classification"},
        {"name": "dili", "category": "toxicity", "type": "binary_classification"},
        {"name": "solubility_aqsoldb", "category": "physicochemical", "type": "regression"},
    ]

    # Test with frozen backbone
    config = {
        "backbone_type": "gine",
        "backbone_config": {
            "in_dim": 14,
            "edge_dim": 6,
            "hidden_dim": 64,
            "num_layers": 2,
        },
        "backbone_hidden_dim": 128,  # GINE pools mean+sum -> 64*2 = 128
        "freeze_backbone": True,
        "head_hidden_dim": 64,
        "tasks": tasks,
        "specialized_tasks": ["caco2_wang", "herg"],
        "use_uncertainty": True,
    }

    model = CategoricalMTLModel(config)

    # Verify backbone freezing
    params = model.parameter_summary()
    assert params["trainable_backbone"] == 0
    assert params["frozen_backbone"] > 0
    assert params["category_heads"] > 0
    assert params["specialized_adapters"] > 0

    # Build small batch
    dm = MultiTaskDataModule(tasks=tasks, synthetic_df=mock_multitask_df, modality="graph")
    dm.prepare_data()
    loader, _, _ = dm.setup_loaders(batch_size=3)
    batch = next(iter(loader))

    # Forward pass
    preds = model(batch)
    assert preds.shape == (3, 5)

    # Compute loss via model helper
    loss = model.compute_loss(preds, batch["labels"], mask=batch["mask"])
    assert loss.item() > 0.0

    # Backward pass only updates heads and adapters
    loss.backward()
    for p in model.backbone.parameters():
        assert p.grad is None

    for head in model.category_heads.values():
        for p in head.parameters():
            if p.requires_grad:
                assert p.grad is not None

    for adapter in model.specialized_adapters.values():
        for p in adapter.parameters():
            if p.requires_grad:
                assert p.grad is not None


def test_threshold_decision_engine():
    engine = ThresholdDecisionEngine()

    # Caco-2
    res_caco_high = engine.evaluate("caco2_wang", -4.8, "regression")
    assert "High Permeability" in res_caco_high["decision"]
    assert res_caco_high["unit"] == "10^-6 cm/s"

    res_caco_low = engine.evaluate("caco2_wang", -6.5, "regression")
    assert "Low Permeability" in res_caco_low["decision"]

    # hERG
    res_herg_safe = engine.evaluate("herg", -2.5, "binary_classification")  # logit -> prob ~ 0.07
    assert res_herg_safe["probability"] < 0.3
    assert "Low Cardiotoxicity" in res_herg_safe["decision"]

    res_herg_toxic = engine.evaluate("herg", 3.0, "binary_classification")  # logit -> prob ~ 0.95
    assert "High Cardiotoxicity" in res_herg_toxic["decision"]

    # DILI
    res_dili_safe = engine.evaluate("dili", 0.1, "binary_classification")
    assert "Safe" in res_dili_safe["decision"]


def test_multitask_inference_pipeline():
    tasks = [
        {"name": "caco2_wang", "category": "absorption", "type": "regression"},
        {"name": "herg", "category": "toxicity", "type": "binary_classification"},
        {"name": "solubility_aqsoldb", "category": "physicochemical", "type": "regression"},
    ]
    config = {
        "backbone_type": "gine",
        "backbone_config": {"in_dim": 14, "edge_dim": 6, "hidden_dim": 32, "num_layers": 2},
        "backbone_hidden_dim": 64,
        "tasks": tasks,
    }
    model = CategoricalMTLModel(config)

    pipeline = MultiTaskInferencePipeline(
        model=model,
        task_configs=tasks,
        modality="graph",
        device="cpu",
    )

    aspirin_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
    result = pipeline.predict(aspirin_smiles)

    assert result["raw_smiles"] == aspirin_smiles
    assert result["smiles"] == "CC(=O)Oc1ccccc1C(=O)O"  # Canonicalized by RDKit
    assert result["elapsed_ms"] >= 0.0

    assert "caco2_wang" in result["predictions"]
    assert "herg" in result["predictions"]
    assert "decision" in result["predictions"]["caco2_wang"]
    assert "decision" in result["predictions"]["herg"]


def test_cli_train_categorical_mtl_dry_run(tmp_path, mock_multitask_df):
    import yaml
    from typer.testing import CliRunner

    from tdc_studio.cli import app
    from tdc_studio.core.registry import DATASETS

    class SyntheticMultiTaskDataModule(MultiTaskDataModule):
        def __init__(self, **kwargs):
            tasks = [
                {"name": "caco2_wang", "category": "absorption", "type": "regression"},
                {"name": "herg", "category": "toxicity", "type": "binary_classification"},
            ]
            super().__init__(tasks=tasks, synthetic_df=mock_multitask_df, **kwargs)

    DATASETS.register("synthetic_mtl_loader")(SyntheticMultiTaskDataModule)

    config_content = {
        "data": {
            "type": "synthetic_mtl_loader",
            "name": "synthetic_mtl",
            "batch_size": 2,
        },
        "model": {
            "type": "categorical_mtl",
            "backbone_type": "gine",
            "backbone_config": {"in_dim": 14, "edge_dim": 6, "hidden_dim": 16, "num_layers": 1},
            "backbone_hidden_dim": 32,
            "head_hidden_dim": 16,
            "tasks": [
                {"name": "caco2_wang", "category": "absorption", "type": "regression"},
                {"name": "herg", "category": "toxicity", "type": "binary_classification"},
            ],
            "specialized_tasks": ["caco2_wang"],
        },
        "lr": 0.001,
        "max_epochs": 1,
        "tracking": {"enabled": False},
    }

    config_file = str(tmp_path / "mtl_config.yaml")
    with open(config_file, "w") as f:
        yaml.dump(config_content, f)

    chk_dir = str(tmp_path / "checkpoints")
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["train", "--config", config_file, "--checkpoint-dir", chk_dir, "--dry-run"],
    )
    assert result.exit_code == 0
    assert "Starting Training Pipeline" in result.output
    assert "Training Pipeline Finished" in result.output

