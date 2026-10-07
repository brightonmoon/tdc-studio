"""Unit tests for DMPNNHurdleModel and HurdleMultiTaskLoss."""

import torch
from torch_geometric.data import Batch, Data

from tdc_studio.core.registry import MODELS
from tdc_studio.models.graph.dmpnn_hurdle import DMPNNHurdleModel
from tdc_studio.models.loss.hurdle_loss import HurdleMultiTaskLoss


def create_dummy_graph_batch(batch_size: int = 4, desc_dim: int = 210):
    """Helper to create dummy molecular graph batch and descriptor matrix."""
    graphs = []
    for _ in range(batch_size):
        num_atoms = 5
        x = torch.randn(num_atoms, 14)  # 14 atom features
        # 4 undirected bonds = 8 directed edges
        edge_index = torch.tensor(
            [
                [0, 1, 1, 0, 1, 2, 2, 1],
                [1, 0, 0, 1, 2, 1, 1, 2],
            ],
            dtype=torch.long,
        )
        edge_attr = torch.randn(8, 6)
        graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))

    batch = Batch.from_data_list(graphs)
    desc = torch.randn(batch_size, desc_dim)
    return {"drug_graph": batch, "descriptors": desc}


def test_dmpnn_hurdle_initialization():
    """Verify registry lookup and model initialization."""
    model_cls = MODELS.get("dmpnn_hurdle")
    assert model_cls is DMPNNHurdleModel

    cfg = {
        "in_dim": 14,
        "edge_dim": 6,
        "hidden_dim": 64,
        "depth": 2,
        "dropout": 0.1,
        "use_descriptors": True,
        "descriptor_dim": 210,
        "high_threshold": 90.0,
        "low_threshold": 70.0,
    }
    model = DMPNNHurdleModel(cfg)

    assert hasattr(model, "gate_head")
    assert hasattr(model, "high_head")
    assert hasattr(model, "low_head")
    assert hasattr(model, "loss_fn")


def test_dmpnn_hurdle_forward_modes():
    """Verify training, evaluation, and dict-returning forward passes."""
    cfg = {
        "in_dim": 14,
        "edge_dim": 6,
        "hidden_dim": 64,
        "depth": 2,
        "descriptor_dim": 210,
    }
    model = DMPNNHurdleModel(cfg)
    batch = create_dummy_graph_batch(batch_size=4, desc_dim=210)

    # 1. Training mode -> Tensor of shape (B, 4)
    model.train()
    out_train = model(batch)
    assert isinstance(out_train, torch.Tensor)
    assert out_train.shape == (4, 4)

    # 2. Evaluation mode -> Tensor of shape (B, 1) (pred_mixture)
    model.eval()
    with torch.no_grad():
        out_eval = model(batch)
    assert isinstance(out_eval, torch.Tensor)
    assert out_eval.shape == (4, 1)

    # 3. Dict mode -> Dictionary of intermediate outputs
    out_dict = model(batch, return_dict=True)
    assert isinstance(out_dict, dict)
    assert "gate_logits" in out_dict
    assert "gate_prob" in out_dict
    assert "pred_high" in out_dict
    assert "pred_low" in out_dict
    assert "pred_mixture" in out_dict
    assert out_dict["gate_prob"].shape == (4, 1)


def test_hurdle_loss_backward():
    """Verify Hurdle loss calculation and gradient backpropagation."""
    loss_fn = HurdleMultiTaskLoss(
        high_threshold=90.0,
        low_threshold=70.0,
        weight_gate=1.0,
        weight_high=0.5,
        weight_low=0.8,
        weight_mixture=1.0,
    )

    B = 6
    preds = {
        "gate_logits": torch.randn(B, 1, requires_grad=True),
        "pred_high": torch.randn(B, 1, requires_grad=True),
        "pred_low": torch.randn(B, 1, requires_grad=True),
    }
    # Diverse targets: some low (<70), mid (70-90), high (>=90)
    targets = torch.tensor([[35.0], [55.0], [78.0], [88.0], [94.0], [98.5]])

    total_loss, metrics = loss_fn(preds, targets)

    assert isinstance(total_loss, torch.Tensor)
    assert total_loss.item() > 0.0
    assert "loss_gate" in metrics
    assert "loss_high" in metrics
    assert "loss_low" in metrics
    assert "loss_mixture" in metrics

    # Test backward pass
    total_loss.backward()
    assert preds["gate_logits"].grad is not None
    assert preds["pred_high"].grad is not None
    assert preds["pred_low"].grad is not None


def test_dmpnn_hurdle_compute_loss_e2e():
    """Verify end-to-end model forward and compute_loss pipeline."""
    cfg = {
        "in_dim": 14,
        "edge_dim": 6,
        "hidden_dim": 64,
        "depth": 2,
        "descriptor_dim": 210,
    }
    model = DMPNNHurdleModel(cfg)
    model.train()

    batch = create_dummy_graph_batch(batch_size=4, desc_dim=210)
    targets = torch.tensor([[45.0], [72.0], [92.0], [96.0]])

    preds = model(batch)
    loss = model.compute_loss(preds, targets)

    assert isinstance(loss, torch.Tensor)
    assert loss.item() > 0.0

    loss.backward()
    # Check that model weights have gradients
    assert model.w_i.weight.grad is not None
    assert model.gate_head[0].weight.grad is not None
    assert model.high_head[0].weight.grad is not None
    assert model.low_head[0].weight.grad is not None
