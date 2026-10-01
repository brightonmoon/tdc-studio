"""Unit and Integration tests for Sprint 3 (P2) & Sprint 4 (P3) DTA Next-Gen components:
- PocketCrossAttentionFusion with 3D coordinate guidance & residue importance
- DualModalDrugEncoder with 2D topology + 3D conformation gating
- ConformalCalibrator split conformal prediction UQ
- DTIInferencePipeline with Conformal UQ
- SelfCorrectingOptimizer with joint DTA potency + ADMET optimization
"""

import math
import tempfile
import numpy as np
import pytest
import torch
import torch.nn as nn

from tdc_studio.core.registry import MODELS
from tdc_studio.models.dti.fusion import PocketCrossAttentionFusion
from tdc_studio.models.dti.dual_modal_encoder import (
    DualModalDrugEncoder,
    SpatialRBFDistanceEncoder,
    generate_3d_coordinates,
)
from tdc_studio.evaluation.conformal import ConformalCalibrator, ConformalPrediction
from tdc_studio.serving.pipeline import DTIInferencePipeline
from tdc_studio.generative.lead_optimizer import SelfCorrectingOptimizer


# ------------------------------------------------------------------------------
# 1. PocketCrossAttentionFusion Tests
# ------------------------------------------------------------------------------

def test_pocket_cross_attention_forward():
    fusion = PocketCrossAttentionFusion({
        "drug_dim": 64,
        "target_dim": 64,
        "hidden_dim": 64,
        "num_heads": 2,
        "out_dim": 1,
    })

    B, L_drug, L_pocket = 2, 16, 32
    h_drug = torch.randn(B, L_drug, 64)
    h_target = torch.randn(B, L_pocket, 64)
    pocket_coords = torch.randn(B, L_pocket, 3) * 10.0
    residue_importance = torch.rand(B, L_pocket)  # pLDDT or SASA

    # Standard forward
    out = fusion(h_drug, h_target)
    assert out.shape == (B, 1)

    # 3D Guided forward with coordinates and residue importance
    out_3d = fusion(
        h_drug=h_drug,
        h_target=h_target,
        pocket_coords=pocket_coords,
        residue_importance=residue_importance,
    )
    assert out_3d.shape == (B, 1)

    # Return attention and XAI metrics
    pred, attn_dict = fusion(
        h_drug=h_drug,
        h_target=h_target,
        pocket_coords=pocket_coords,
        residue_importance=residue_importance,
        return_attention=True,
    )
    assert pred.shape == (B, 1)
    assert "attn_d2t" in attn_dict
    assert "attn_t2d" in attn_dict
    assert "contact_map" in attn_dict
    assert "pocket_residue_importance" in attn_dict
    assert attn_dict["contact_map"].shape == (B, L_drug, L_pocket)
    assert attn_dict["pocket_residue_importance"].shape == (B, L_pocket)


def test_pocket_cross_attention_registry():
    model = MODELS.build({
        "type": "pocket_cross_attention",
        "drug_dim": 32,
        "target_dim": 32,
        "hidden_dim": 32,
    })
    assert isinstance(model, PocketCrossAttentionFusion)


# ------------------------------------------------------------------------------
# 2. DualModalDrugEncoder Tests
# ------------------------------------------------------------------------------

def test_dual_modal_drug_encoder():
    encoder = DualModalDrugEncoder({
        "hidden_dim": 64,
        "out_dim": 64,
        "use_3d_coordinates": True,
    })

    smiles_list = ["CCO", "c1ccccc1O", "CC(=O)Oc1ccccc1C(=O)O"]
    batch = {"drug_smiles_str": smiles_list}

    h_drug = encoder.extract_features(batch)
    assert h_drug.shape == (3, 64)

    # Sequence return mode
    h_seq = encoder.extract_features(batch, return_sequence=True)
    assert h_seq.shape == (3, 1, 64)


def test_spatial_rbf_encoder():
    rbf_enc = SpatialRBFDistanceEncoder(num_rbf=16, hidden_dim=32)
    coords = torch.randn(10, 3) * 5.0
    feat = rbf_enc(coords)
    assert feat.shape == (32,)


# ------------------------------------------------------------------------------
# 3. ConformalCalibrator Tests
# ------------------------------------------------------------------------------

def test_conformal_calibrator_exact_coverage():
    np.random.seed(42)
    N = 1000
    y_true = np.random.normal(7.0, 1.5, size=N)
    # Add Gaussian noise
    y_pred = y_true + np.random.normal(0.0, 0.5, size=N)

    calibrator = ConformalCalibrator(alpha=0.05)
    q_hat = calibrator.calibrate(y_true, y_pred)

    assert calibrator.calibrated is True
    assert q_hat > 0.0
    # For normal(0, 0.5), 95% quantile is ~1.96 * 0.5 = ~0.98
    assert 0.85 < q_hat < 1.15

    # Check coverage on independent test set
    M = 2000
    y_test_true = np.random.normal(7.0, 1.5, size=M)
    y_test_pred = y_test_true + np.random.normal(0.0, 0.5, size=M)

    uq_list = calibrator.predict_interval(y_test_pred)
    assert len(uq_list) == M

    covered = sum(
        (uq.lower <= true_val <= uq.upper)
        for uq, true_val in zip(uq_list, y_test_true)
    )
    empirical_coverage = covered / M
    # Should satisfy empirical coverage >= 1 - alpha - small tolerance
    assert empirical_coverage >= 0.93


def test_conformal_calibrator_save_load(tmp_path):
    calibrator = ConformalCalibrator(alpha=0.10)
    y_true = [5.0, 6.0, 7.0, 8.0]
    y_pred = [5.2, 5.9, 7.3, 7.8]
    calibrator.calibrate(y_true, y_pred)

    save_file = tmp_path / "conformal_params.json"
    calibrator.save(save_file)
    assert save_file.is_file()

    loaded = ConformalCalibrator.load(save_file)
    assert loaded.calibrated is True
    assert loaded.alpha == 0.10
    assert loaded.q_hat == calibrator.q_hat
    assert loaded.cal_size == 4


# ------------------------------------------------------------------------------
# 4. DTIInferencePipeline with Conformal UQ
# ------------------------------------------------------------------------------

def test_dti_pipeline_with_conformal_uq():
    class DummyDTA(nn.Module):
        def forward(self, batch, **kwargs):
            B = len(batch.get("drug_smiles_str", [1]))
            return torch.tensor([[7.5]] * B)

    pipeline = DTIInferencePipeline(
        model=DummyDTA(),
        device="cpu",
    )

    res = pipeline.predict_affinity(
        smiles_list=["CCO", "c1ccccc1"],
        target_seqs=["MKTAY", "MSHHW"],
        return_uncertainty=True,
    )

    assert "predictions_pkd" in res
    assert len(res["predictions_pkd"]) == 2
    assert "conformal_lower_95" in res
    assert "conformal_upper_95" in res
    assert "confidence_interval_width" in res
    assert "is_in_domain" in res

    assert len(res["conformal_lower_95"]) == 2
    assert res["conformal_lower_95"][0] < res["predictions_pkd"][0] < res["conformal_upper_95"][0]
    assert res["is_in_domain"] == [True, True]


# ------------------------------------------------------------------------------
# 5. Joint Lead Optimization (DTA + ADMET)
# ------------------------------------------------------------------------------

def test_joint_lead_optimization():
    class MockDTAPipeline:
        def predict_affinity(self, smiles_list, target_seqs, **kwargs):
            return {"predictions_pkd": [8.2 for _ in smiles_list]}

    optimizer = SelfCorrectingOptimizer(
        device="cpu",
        sa_threshold=4.5,
        dti_pipeline=MockDTAPipeline(),
    )

    report = optimizer.optimize(
        smiles="O=[N+]([O-])c1ccccc1",
        target_liability="ames",
        target_seq="MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQNDFSFLDPV",
        weight_admet=1.0,
        weight_dta=0.5,
        max_candidates=2,
        steps=5,
    )

    assert report.target_protein_sequence is not None
    assert report.parent_dta_pkd == 8.2
    assert len(report.top_candidates) > 0
    top1 = report.top_candidates[0]
    assert top1.parent_dta_pkd == 8.2
    assert top1.candidate_dta_pkd == 8.2
    assert top1.dta_delta == 0.0
