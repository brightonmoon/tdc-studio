"""Tests for Data pipeline: transforms, collator, and data modules (zero heavy training)."""

import torch

from tdc_studio.data.collate import molecule_collate_fn
from tdc_studio.data.multi_pred import DTADataModule
from tdc_studio.data.single_pred import ADMETDataModule
from tdc_studio.data.transforms import SequenceTokenizer, SmilesToGraphTransform


def test_smiles_to_graph_transform():
    transform = SmilesToGraphTransform()
    # Aspirin
    data = transform("CC(=O)OC1=CC=CC=C1C(=O)O")
    assert data is not None
    assert data.x.size(1) == 14  # 10 atom types + 4 properties
    assert data.edge_index.size(0) == 2
    assert data.num_nodes > 0

    # Invalid SMILES returns None
    invalid = transform("INVALID_SMILES_STRING")
    assert invalid is None


def test_sequence_tokenizer():
    tok = SequenceTokenizer(max_length=10)
    seq = "MKWVTF"
    tokens = tok(seq)
    assert isinstance(tokens, torch.Tensor)
    assert tokens.dtype == torch.long
    assert len(tokens) == 6


def test_molecule_collate_fn():
    transform = SmilesToGraphTransform()
    g1 = transform("CCO")  # Ethanol
    g2 = transform("CCN")  # Ethylamine

    items = [
        {"drug_graph": g1, "label": 1.0},
        {"drug_graph": g2, "label": 2.0},
    ]
    batch = molecule_collate_fn(items)
    assert "drug_graph" in batch
    assert "labels" in batch
    assert batch["labels"].shape == torch.Size([2])
    assert batch["drug_graph"].num_graphs == 2


def test_admet_datamodule_with_synthetic_data(dummy_smiles_df):
    dm = ADMETDataModule(dataset_name="toy_admet", synthetic_df=dummy_smiles_df)
    dm.prepare_data()
    assert dm.is_prepared
    assert "train" in dm.splits
    assert len(dm.splits["train"]) > 0

    train_loader, val_loader, test_loader = dm.setup_loaders(batch_size=2)
    batch = next(iter(train_loader))
    assert "drug_graph" in batch
    assert "labels" in batch
    assert batch["labels"].shape[0] <= 2


def test_dta_datamodule_with_synthetic_data(dummy_dta_df):
    dm = DTADataModule(dataset_name="toy_dta", synthetic_df=dummy_dta_df)
    dm.prepare_data()
    assert dm.is_prepared

    train_loader, _, _ = dm.setup_loaders(batch_size=2)
    batch = next(iter(train_loader))
    assert "drug_graph" in batch
    assert "target_seq" in batch
    assert "labels" in batch


def test_dta_dual_cold_split(dummy_dta_df):
    """Verify that dual_cold split enforces disjoint drugs AND disjoint targets between train and test."""
    dm = DTADataModule(
        dataset_name="toy_dual_cold",
        split_type="dual_cold",
        synthetic_df=dummy_dta_df,
        frac=[0.5, 0.25, 0.25],
        seed=42,
    )
    dm.prepare_data()
    assert dm.is_prepared

    train_drugs = set(dm.splits["train"]["Drug"].unique())
    test_drugs = set(dm.splits["test"]["Drug"].unique())
    train_targets = set(dm.splits["train"]["Target"].unique())
    test_targets = set(dm.splits["test"]["Target"].unique())

    # Strictly zero overlap in both modalities
    assert len(train_drugs.intersection(test_drugs)) == 0
    assert len(train_targets.intersection(test_targets)) == 0


def test_dta_kiba_auto_scaling(dummy_dta_df):
    """Verify KIBA dataset disables log_transform by default."""
    dm = DTADataModule(dataset_name="KIBA", synthetic_df=dummy_dta_df)
    dm.prepare_data()
    assert dm.log_transform is False
