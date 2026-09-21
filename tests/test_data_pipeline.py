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
