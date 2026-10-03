"""Zero-training unit tests for CrossAttentionFusion and GraphDTAModel integration."""

import torch
from torch_geometric.data import Batch, Data

from tdc_studio.models.dti.dta_model import GraphDTAModel
from tdc_studio.models.dti.fusion import CrossAttentionFusion


def test_cross_attention_fusion_forward_2d():
    """Test CrossAttentionFusion with pooled 2D inputs [B, D]."""
    cfg = {
        "drug_dim": 64,
        "target_dim": 64,
        "hidden_dim": 32,
        "num_heads": 2,
        "out_dim": 1,
    }
    fusion = CrossAttentionFusion(cfg)
    h_drug = torch.randn(4, 64)
    h_target = torch.randn(4, 64)

    out = fusion(h_drug, h_target)
    assert out.shape == (4, 1)
    assert not torch.isnan(out).any()


def test_cross_attention_fusion_forward_3d():
    """Test CrossAttentionFusion with sequence 3D inputs [B, L, D]."""
    cfg = {
        "drug_dim": 32,
        "target_dim": 48,
        "hidden_dim": 64,
        "num_heads": 4,
        "out_dim": 1,
    }
    fusion = CrossAttentionFusion(cfg)
    h_drug = torch.randn(2, 12, 32)  # 12 drug tokens
    h_target = torch.randn(2, 40, 48)  # 40 amino acid residues

    out = fusion(h_drug, h_target)
    assert out.shape == (2, 1)
    assert not torch.isnan(out).any()


def test_cross_attention_fusion_attention_weights():
    """Test extraction of bidirectional attention maps for XAI interpretability."""
    cfg = {
        "drug_dim": 32,
        "target_dim": 32,
        "hidden_dim": 32,
        "num_heads": 2,
        "out_dim": 1,
    }
    fusion = CrossAttentionFusion(cfg)
    fusion.eval()
    h_drug = torch.randn(2, 5, 32)  # 5 drug tokens
    h_target = torch.randn(2, 15, 32)  # 15 protein residues

    out, attn_dict = fusion(h_drug, h_target, return_attention=True)
    assert out.shape == (2, 1)
    assert "attn_d2t" in attn_dict
    assert "attn_t2d" in attn_dict

    # Check attention weights shape: [B, num_heads, Q_len, K_len]
    attn_d2t = attn_dict["attn_d2t"]
    attn_t2d = attn_dict["attn_t2d"]
    assert attn_d2t.shape == (2, 2, 5, 15)
    assert attn_t2d.shape == (2, 2, 15, 5)

    # Attention weights must sum to 1 over key dimension
    assert torch.allclose(attn_d2t.sum(dim=-1), torch.ones(2, 2, 5), atol=1e-4)
    assert torch.allclose(attn_t2d.sum(dim=-1), torch.ones(2, 2, 15), atol=1e-4)


def test_graph_dta_model_with_cross_attention():
    """Test end-to-end forward and backward pass of GraphDTAModel with CrossAttentionFusion."""
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
        },
    }
    model = GraphDTAModel(model_cfg)

    # Construct minimal mock batch
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
    batch_graph = Batch.from_data_list([g1, g2])

    batch = {
        "drug_graph": batch_graph,
        "target_seq": torch.randint(1, 25, (2, 20)),
    }

    # 1. Forward
    preds = model(batch)
    assert preds.shape == (2, 1)

    # 2. Forward with attention
    preds_attn, attn_dict = model(batch, return_attention=True)
    assert preds_attn.shape == (2, 1)
    assert "attn_d2t" in attn_dict

    # 3. Backward
    targets = torch.tensor([5.0, 7.5])
    loss = model.compute_loss(preds, targets)
    loss.backward()

    # Check gradients
    assert model.fusion.proj_drug.weight.grad is not None
    assert model.fusion.head[-1].weight.grad is not None
