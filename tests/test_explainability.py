"""Unit tests for Explainable AI (XAI) and Bioisostere Recommender modules."""

import numpy as np
import torch
import torch.nn as nn

from tdc_studio.explainability.attribution import MolecularExplainer
from tdc_studio.explainability.bioisostere import BioisostereRecommender
from tdc_studio.explainability.visualizer import AttributionVisualizer


class MockGNNModel(nn.Module):
    """Simple linear mock model taking batch['drug_graph']."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(14, 1)

    def forward(self, batch):
        g = batch["drug_graph"]
        # Global mean pool manually
        batch_idx = g.batch
        out_list = []
        for i in range(int(batch_idx.max().item()) + 1):
            mask = batch_idx == i
            node_feats = g.x[mask]
            pooled = node_feats.mean(dim=0, keepdim=True)
            out_list.append(self.lin(pooled))
        return torch.cat(out_list, dim=0)


def test_integrated_gradients_molecular_explainer():
    model = MockGNNModel()
    explainer = MolecularExplainer(model, steps=10, device="cpu")

    smiles = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin (13 heavy atoms)
    res = explainer.attribute(smiles, target_idx=0)

    assert "atom_attributions" in res
    assert "normalized_attributions" in res
    assert res["num_atoms"] == 13
    assert len(res["atom_attributions"]) == 13
    assert len(res["normalized_attributions"]) == 13

    norm_arr = np.array(res["normalized_attributions"])
    assert np.all(norm_arr >= -1.0 - 1e-5)
    assert np.all(norm_arr <= 1.0 + 1e-5)
    assert isinstance(res["predicted_score"], float)


def test_attribution_visualizer_svg_and_data_uri():
    vis = AttributionVisualizer(width=300, height=300)
    smiles = "c1ccccc1"
    weights = [0.5, -0.2, 0.1, 0.0, 0.8, -0.6]

    svg = vis.render_svg(smiles, weights, legend="Benzene Test")
    assert "<svg" in svg
    assert "</svg>" in svg

    data_uri = vis.render_data_uri(smiles, weights)
    assert data_uri.startswith("data:image/svg+xml;base64,")


def test_bioisostere_recommender():
    recommender = BioisostereRecommender()

    # 1. Test Carboxylic acid replacement on Aspirin
    aspirin = "CC(=O)Oc1ccccc1C(=O)O"
    cooh_suggestions = recommender.recommend(aspirin, liability_focus="dili")
    assert len(cooh_suggestions) > 0
    first = cooh_suggestions[0]
    assert "tetrazole" in first["transformation_name"] or "sulfonamide" in first["transformation_name"]
    assert first["modified_smiles"] != aspirin

    # 2. Test Nitro replacement on Nitrobenzene
    nitrobenzene = "c1ccccc1[N+](=O)[O-]"
    nitro_suggestions = recommender.recommend(nitrobenzene, liability_focus="ames")
    assert len(nitro_suggestions) > 0
    rule_names = [s["transformation_name"] for s in nitro_suggestions]
    assert any("nitro_to" in r for r in rule_names)
