"""Unit tests for TargetAttentionExplainer."""

import pytest
import torch
import torch.nn as nn

from tdc_studio.explainability.target_attention import (
    TargetAttentionExplainer,
    TargetAttentionResult,
)


def test_target_attention_heuristic_fallback():
    """Test target residue attention scoring with heuristic fallback."""
    explainer = TargetAttentionExplainer(model=None)

    smiles = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
    seq = "MKTLLILTILAMAFAYFGHI"  # 20 AA peptide

    result = explainer.explain(smiles, seq, target_name="COX-1", top_k=5, generate_svg=True)

    assert isinstance(result, TargetAttentionResult)
    assert result.smiles == smiles
    assert result.target_name == "COX-1"
    assert result.sequence_length == len(seq)
    assert len(result.residue_contributions) == len(seq)
    assert len(result.top_hotspot_residues) == 5

    # Check first residue
    first_res = result.residue_contributions[0]
    assert first_res.position == 1
    assert first_res.amino_acid == "M"
    assert 0.0 <= first_res.score <= 1.0

    # Check SVG generation
    assert result.svg_chart is not None
    assert "<svg" in result.svg_chart
    assert "</svg>" in result.svg_chart

    # Check to_dict serialization
    d = result.to_dict()
    assert d["target_name"] == "COX-1"
    assert "top_hotspot_residues" in d
    assert len(d["top_hotspot_residues"]) == 5
    assert d["svg_data_uri"].startswith("data:image/svg+xml;base64,")


def test_target_attention_empty_sequence():
    """Test error handling when an empty sequence is provided."""
    explainer = TargetAttentionExplainer()
    with pytest.raises(ValueError, match="Target sequence cannot be empty"):
        explainer.explain("c1ccccc1", "")


def test_target_attention_with_mock_model():
    """Test extracting contact matrix from a mock model supporting return_attention=True."""

    class MockDTAModel(nn.Module):
        def forward(self, batch, return_attention=False):
            b_size = 1
            l_drug = 10
            l_target = 15
            affinity = torch.tensor([[8.5]])
            # Synthetic contact map: make residue index 4 (5th residue) have very high attention
            contact = torch.zeros(b_size, l_drug, l_target)
            contact[:, :, 4] = 5.0
            contact[:, :, 9] = 3.0
            return affinity, {"contact_map": contact}

    mock_model = MockDTAModel()
    explainer = TargetAttentionExplainer(model=mock_model)

    smiles = "c1ccccc1"
    seq = "ACDEFGHIKLMNPQR"  # 15 AA

    result = explainer.explain(smiles, seq, top_k=3, generate_svg=False)

    assert result.predicted_affinity == 8.5
    # The 5th residue (F, index 4) should be the #1 top hotspot
    top_1 = result.top_hotspot_residues[0]
    assert top_1.position == 5
    assert top_1.amino_acid == "F"
    assert top_1.score == 1.0  # max normalized
