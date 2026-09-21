"""Tests for Model architectures (forward shape and loss check, zero heavy training)."""

import torch

from tdc_studio.data.collate import molecule_collate_fn
from tdc_studio.data.transforms import SequenceTokenizer, SmilesToGraphTransform
from tdc_studio.models.graph.graph_transformer import (
    GraphTransformerDTAModel,
    GraphTransformerModel,
)
from tdc_studio.models.sequence.transformer import SequenceTransformerModel


def test_graph_transformer_forward_and_loss():
    transform = SmilesToGraphTransform()
    g1 = transform("CC(=O)O")
    g2 = transform("CCN")

    batch = molecule_collate_fn(
        [
            {"drug_graph": g1, "label": 1.5},
            {"drug_graph": g2, "label": 2.5},
        ]
    )

    config = {"in_dim": 14, "hidden_dim": 32, "num_layers": 2, "task_type": "regression"}
    model = GraphTransformerModel(config)

    preds = model(batch)
    assert preds.shape == torch.Size([2, 1])

    loss = model.compute_loss(preds, batch["labels"])
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert not torch.isnan(loss)


def test_graph_transformer_dta_forward():
    transform = SmilesToGraphTransform()
    tok = SequenceTokenizer()
    g1 = transform("CCO")
    t1 = tok("MKWVTF")

    batch = molecule_collate_fn([{"drug_graph": g1, "target_seq": t1, "label": 0.5}])
    config = {"in_dim": 14, "hidden_dim": 32, "vocab_size": 30, "task_type": "regression"}
    model = GraphTransformerDTAModel(config)

    preds = model(batch)
    assert preds.shape == torch.Size([1, 1])


def test_sequence_transformer_forward():
    tok = SequenceTokenizer()
    seqs = torch.stack([tok("MKWVTF"), tok("MAGFLK")])
    batch = {"target_seq": seqs, "labels": torch.tensor([1.0, 2.0])}

    config = {"vocab_size": 30, "hidden_dim": 32, "nhead": 2, "num_layers": 1}
    model = SequenceTransformerModel(config)

    preds = model(batch)
    assert preds.shape == torch.Size([2, 1])
