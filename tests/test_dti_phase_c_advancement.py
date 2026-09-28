"""Zero-training unit tests for DTI Phase C Advancement: Task F-1, F-2, F-3."""

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch, Data

from tdc_studio.data.multi_pred import (
    MaskedMSELoss,
    MultiAffinityDTADataModule,
)
from tdc_studio.models.dti.dta_model import GraphDTAModel
from tdc_studio.models.dti.fusion import CrossAttentionFusion


def test_task_f1_contact_map_extraction():
    """Task F-1: Test CrossAttentionFusion with 3D sequence tensors and 2D contact map."""
    cfg = {
        "drug_dim": 32,
        "target_dim": 48,
        "hidden_dim": 64,
        "num_heads": 4,
        "out_dim": 1,
    }
    fusion = CrossAttentionFusion(cfg)
    fusion.eval()

    # Batch of 2 samples: drug length 15 tokens, target length 50 residues
    h_drug = torch.randn(2, 15, 32)
    h_target = torch.randn(2, 50, 48)

    pred, attn_dict = fusion(h_drug, h_target, return_attention=True)

    # 1. Output shape
    assert pred.shape == (2, 1)

    # 2. Contact map shape: [B, L_drug, L_target]
    assert "contact_map" in attn_dict
    contact_map = attn_dict["contact_map"]
    assert contact_map.shape == (2, 15, 50)
    assert not torch.isnan(contact_map).any()

    # 3. Sum over target dimension should equal 1 for each drug token
    assert torch.allclose(contact_map.sum(dim=-1), torch.ones(2, 15), atol=1e-4)


def test_task_f1_graph_dta_full_sequence_forward():
    """Task F-1: Test GraphDTAModel forward with return_sequence and contact map."""
    model_cfg = {
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
            "kernel_sizes": [3],
        },
        "fusion": {
            "type": "cross_attention",
            "hidden_dim": 32,
            "num_heads": 2,
            "out_dim": 1,
        },
    }
    model = GraphDTAModel(model_cfg)

    g1 = Data(
        x=torch.randn(4, 14),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
        edge_attr=torch.randn(3, 6),
    )
    g2 = Data(
        x=torch.randn(3, 14),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        edge_attr=torch.randn(2, 6),
    )
    batch = {
        "drug_graph": Batch.from_data_list([g1, g2]),
        "target_seq": torch.randint(1, 25, (2, 30)),
    }

    preds, attn_dict = model(batch, return_attention=True)
    assert preds.shape == (2, 1)
    assert "contact_map" in attn_dict


def test_task_f2_masked_loss_and_gradients():
    """Task F-2 & F-3: Test MaskedMSELoss handles missing labels and backprops correctly."""
    loss_fn = MaskedMSELoss()

    # 3 samples, 3 tasks (Kd, Ki, IC50)
    preds = torch.tensor(
        [[1.0, 2.0, 3.0], [0.5, 1.5, 2.5], [2.0, 2.0, 2.0]],
        requires_grad=True,
    )
    targets = torch.tensor(
        [[1.0, 0.0, 3.0], [0.0, 1.5, 0.0], [2.0, 3.0, 2.0]],
    )
    # Mask: 1=valid, 0=missing
    mask = torch.tensor(
        [[True, False, True], [False, True, False], [True, True, True]],
        dtype=torch.bool,
    )

    loss = loss_fn(preds, targets, mask=mask)
    assert torch.isfinite(loss)
    assert loss.item() > 0.0

    # Backpropagation test
    loss.backward()
    assert preds.grad is not None
    # For masked-out entries (e.g. sample 0 task 1, sample 1 task 0 & 2), grad must be 0
    assert preds.grad[0, 1].item() == 0.0
    assert preds.grad[1, 0].item() == 0.0
    assert preds.grad[1, 2].item() == 0.0
    # For valid entries with diff > 0, grad must be non-zero
    assert preds.grad[2, 1].item() != 0.0


def test_task_f3_multi_affinity_datamodule_synthetic():
    """Task F-3: Test MultiAffinityDTADataModule with synthetic data and leakage-free split."""
    # Synthetic DataFrame with valid SMILES
    records = []
    drugs = [f"CC(=O)NC{'C' * (i % 6 + 1)}" for i in range(20)]
    targets = [f"MSHHW{'A' * (j % 4 + 1)}" for j in range(5)]

    for d in drugs:
        for t in targets:
            records.append(
                {
                    "Drug_ID": f"d_{d}",
                    "Drug": d,
                    "Target_ID": f"t_{t}",
                    "Target": t,
                    "Kd": np.random.uniform(1.0, 100.0) if np.random.rand() > 0.4 else np.nan,
                    "Ki": np.random.uniform(5.0, 200.0) if np.random.rand() > 0.4 else np.nan,
                    "IC50": np.random.uniform(10.0, 500.0) if np.random.rand() > 0.3 else np.nan,
                }
            )
    synthetic_df = pd.DataFrame(records)

    dm = MultiAffinityDTADataModule(
        synthetic_df=synthetic_df,
        frac=[0.7, 0.1, 0.2],
        seed=42,
    )
    dm.prepare_data()

    # 1. Leakage audit: Test drugs must never overlap with train drugs
    train_drugs = set(dm.splits["train"]["Drug"].unique())
    test_drugs = set(dm.splits["test"]["Drug"].unique())
    assert len(train_drugs.intersection(test_drugs)) == 0

    # 2. Check loaders
    train_loader, val_loader, test_loader = dm.setup_loaders(batch_size=8)
    batch = next(iter(train_loader))

    assert "drug_graph" in batch
    assert "target_seq" in batch
    assert "labels" in batch
    assert "mask" in batch
    assert batch["labels"].shape[1] == 3
    assert batch["mask"].shape[1] == 3


def test_task_f3_multi_head_graph_dta_model():
    """Task F-3: Test GraphDTAModel with out_dim=3 for multi-task DTA."""
    model_cfg = {
        "type": "graph_dta",
        "out_dim": 3,
        "drug_encoder": {
            "type": "gine",
            "in_dim": 14,
            "hidden_dim": 16,
            "num_layers": 1,
        },
        "target_encoder": {
            "type": "protein_cnn",
            "out_dim": 16,
            "kernel_sizes": [3],
        },
        "fusion": {
            "type": "cross_attention",
            "hidden_dim": 32,
            "num_heads": 2,
            "out_dim": 3,
        },
    }
    model = GraphDTAModel(model_cfg)

    g1 = Data(
        x=torch.randn(3, 14),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        edge_attr=torch.randn(2, 6),
    )
    g2 = Data(
        x=torch.randn(2, 14),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        edge_attr=torch.randn(2, 6),
    )
    batch = {
        "drug_graph": Batch.from_data_list([g1, g2]),
        "target_seq": torch.randint(1, 25, (2, 20)),
        "labels": torch.tensor([[1.0, 2.0, 3.0], [0.0, 1.5, 0.0]]),
        "mask": torch.tensor([[True, True, True], [False, True, False]]),
    }

    preds = model(batch)
    assert preds.shape == (2, 3)

    loss = model.compute_loss(preds, batch["labels"], mask=batch["mask"])
    assert torch.isfinite(loss)
    loss.backward()
    assert model.fusion.head[-1].weight.grad is not None
