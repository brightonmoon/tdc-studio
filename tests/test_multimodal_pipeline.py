"""Tests for Multi-modal pipeline: Enhanced Graph with Edge attributes, SMILES Tokenizer, Fingerprints, and ToxDataModule."""

import torch

from tdc_studio.data.collate import molecule_collate_fn
from tdc_studio.data.single_pred import ToxDataModule
from tdc_studio.data.transforms import (
    MorganFingerprintTransform,
    SmilesToGraphTransform,
    SmilesTokenizer,
)
from tdc_studio.models.fingerprint.mlp import MLPBaselineModel
from tdc_studio.models.graph.gine import GINEModel
from tdc_studio.models.graph.graph_transformer import GraphTransformerModel
from tdc_studio.models.sequence.transformer import SequenceTransformerModel


def test_smiles_tokenizer():
    tokenizer = SmilesTokenizer(max_length=64)
    smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin

    tokens = tokenizer.tokenize(smiles)
    assert len(tokens) > 0
    assert "C" in tokens
    assert "=" in tokens
    assert "(" in tokens

    # Tokenize complex atom brackets and double digits
    complex_smiles = "[nH]1c(=O)[nH]c2c1c(=O)n(C)c(=O)n2C"
    c_tokens = tokenizer.tokenize(complex_smiles)
    assert "[nH]" in c_tokens

    # Encode to tensor
    tensor = tokenizer.encode(smiles)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.ndim == 1
    assert tensor[0] == tokenizer.cls_token_id
    assert tensor[-1] == tokenizer.sep_token_id

    # Decode back
    decoded = tokenizer.decode(tensor)
    assert decoded == smiles


def test_morgan_fingerprint_transform():
    transform = MorganFingerprintTransform(radius=2, n_bits=2048)
    smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"

    fp = transform(smiles)
    assert fp is not None
    assert isinstance(fp, torch.Tensor)
    assert fp.shape == torch.Size([2048])
    assert fp.sum() > 0  # Should have set bits


def test_smiles_to_graph_with_edge_features():
    transform = SmilesToGraphTransform(extended=False)
    data = transform("CC(=O)O")
    assert data is not None
    assert data.x.size(1) == 14
    assert hasattr(data, "edge_attr")
    assert data.edge_attr.size(1) == 6  # 4 bond types + conjugated + in_ring
    assert data.edge_attr.size(0) == data.edge_index.size(1)

    # Extended mode
    ext_transform = SmilesToGraphTransform(extended=True)
    ext_data = ext_transform("CC(=O)O")
    assert ext_data.x.size(1) == 37


def test_gine_model_forward():
    transform = SmilesToGraphTransform(extended=False)
    g1 = transform("CC(=O)O")
    g2 = transform("CCN")

    batch = molecule_collate_fn(
        [
            {"drug_graph": g1, "label": 1.5},
            {"drug_graph": g2, "label": 2.5},
        ]
    )

    config = {
        "in_dim": 14,
        "edge_dim": 6,
        "hidden_dim": 32,
        "num_layers": 2,
        "task_type": "regression",
    }
    model = GINEModel(config)

    preds = model(batch)
    assert preds.shape == torch.Size([2, 1])

    loss = model.compute_loss(preds, batch["labels"])
    assert not torch.isnan(loss)


def test_graph_transformer_with_transformer_conv():
    transform = SmilesToGraphTransform()
    g1 = transform("CCO")
    g2 = transform("c1ccccc1")

    batch = molecule_collate_fn(
        [
            {"drug_graph": g1, "label": 0.5},
            {"drug_graph": g2, "label": 1.0},
        ]
    )

    config = {
        "in_dim": 14,
        "edge_dim": 6,
        "hidden_dim": 32,
        "num_layers": 2,
        "heads": 2,
        "use_transformer_conv": True,
        "task_type": "regression",
    }
    model = GraphTransformerModel(config)

    preds = model(batch)
    assert preds.shape == torch.Size([2, 1])


def test_mlp_baseline_model():
    fp1 = torch.zeros(2048)
    fp1[10] = 1.0
    fp2 = torch.zeros(2048)
    fp2[20] = 1.0

    batch = molecule_collate_fn(
        [
            {"fingerprint": fp1, "label": 1.0},
            {"fingerprint": fp2, "label": 0.0},
        ]
    )
    assert "fingerprint" in batch
    assert batch["fingerprint"].shape == torch.Size([2, 2048])

    config = {
        "in_dim": 2048,
        "hidden_dim": 64,
        "num_layers": 2,
        "task_type": "binary_classification",
    }
    model = MLPBaselineModel(config)

    preds = model(batch)
    assert preds.shape == torch.Size([2, 1])


def test_sequence_transformer_with_smiles_tokenizer():
    tokenizer = SmilesTokenizer(max_length=32)
    s1 = tokenizer("CCO")
    s2 = tokenizer("CCN")

    batch = molecule_collate_fn(
        [
            {"smiles_seq": s1, "label": 1.0},
            {"smiles_seq": s2, "label": 0.0},
        ]
    )
    assert "smiles_seq" in batch
    assert batch["smiles_seq"].ndim == 2

    config = {
        "vocab_size": tokenizer.vocab_size + 10,
        "hidden_dim": 32,
        "nhead": 2,
        "num_layers": 1,
        "task_type": "binary_classification",
    }
    model = SequenceTransformerModel(config)
    preds = model(batch)
    assert preds.shape == torch.Size([2, 1])


def test_tox_datamodule_all_modalities(dummy_smiles_df):
    # 1. Graph modality
    dm_graph = ToxDataModule(
        dataset_name="toy_herg", synthetic_df=dummy_smiles_df, modality="graph"
    )
    dm_graph.prepare_data()
    train_loader, _, _ = dm_graph.setup_loaders(batch_size=2)
    batch_g = next(iter(train_loader))
    assert "drug_graph" in batch_g

    # 2. Sequence modality
    dm_seq = ToxDataModule(
        dataset_name="toy_herg", synthetic_df=dummy_smiles_df, modality="sequence"
    )
    dm_seq.prepare_data()
    train_loader_seq, _, _ = dm_seq.setup_loaders(batch_size=2)
    batch_s = next(iter(train_loader_seq))
    assert "smiles_seq" in batch_s

    # 3. Fingerprint modality
    dm_fp = ToxDataModule(
        dataset_name="toy_herg", synthetic_df=dummy_smiles_df, modality="fingerprint"
    )
    dm_fp.prepare_data()
    train_loader_fp, _, _ = dm_fp.setup_loaders(batch_size=2)
    batch_fp = next(iter(train_loader_fp))
    assert "fingerprint" in batch_fp
