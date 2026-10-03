"""Tests for Conformal Prediction Uncertainty Quantification module."""

import numpy as np

from tdc_studio.uncertainty.conformal import (
    ConformalADMETShield,
    ConformalClassifier,
    ConformalRegressor,
)


def test_conformal_classifier_calibration_and_prediction():
    """Verify split conformal classifier calibration and prediction set construction."""
    rng = np.random.default_rng(42)

    # 1. Calibration on synthetic validation set (n=200)
    true_labels = rng.integers(0, 2, size=200)
    # well-calibrated probabilities with some noise
    probs_pos = np.where(
        true_labels == 1, rng.uniform(0.6, 0.99, size=200), rng.uniform(0.01, 0.4, size=200)
    )

    clf = ConformalClassifier(default_alpha=0.10)
    q = clf.calibrate(probs_pos, true_labels, alpha=0.10)

    assert 0.0 < q < 1.0

    # 2. Test predictions with extreme and ambiguous probabilities
    # High confidence non-toxic (prob ~ 0.02)
    res_safe = clf.predict_set("ames_test", prob_pos=0.02)
    assert res_safe.prediction_set == [0]
    assert res_safe.is_singleton is True
    assert "Safe" in res_safe.uncertainty_tier

    # High confidence toxic (prob ~ 0.98)
    res_toxic = clf.predict_set("herg_test", prob_pos=0.98)
    assert res_toxic.prediction_set == [1]
    assert res_toxic.is_singleton is True
    assert "Liability" in res_toxic.uncertainty_tier

    # Ambiguous / uncertain (prob ~ 0.50)
    res_ambiguous = clf.predict_set("dili_test", prob_pos=0.50)
    assert res_ambiguous.prediction_set == [0, 1]
    assert res_ambiguous.is_singleton is False
    assert "Uncertain" in res_ambiguous.uncertainty_tier


def test_conformal_regressor_calibration_and_intervals():
    """Verify split conformal regressor calibration and interval prediction."""
    rng = np.random.default_rng(123)

    # Synthetic predictions and targets with Gaussian noise std=0.25
    targets = rng.uniform(-1.0, 3.0, size=200)
    preds = targets + rng.normal(0, 0.25, size=200)

    reg = ConformalRegressor(default_alpha=0.10)
    q_margin = reg.calibrate(preds, targets, alpha=0.10)

    assert q_margin > 0.0
    # For std=0.25 normal, 90% quantile is approx 1.645 * 0.25 ~ 0.41
    assert 0.25 < q_margin < 0.70

    # Test interval prediction
    point_pred = 1.50
    res = reg.predict_interval("vdss_test", point_prediction=point_pred, unit="L/kg")

    assert res.point_prediction == point_pred
    assert res.lower_bound == round(point_pred - q_margin, 4)
    assert res.upper_bound == round(point_pred + q_margin, 4)
    assert res.margin_q == round(q_margin, 4)
    assert res.confidence_level == 0.90


def test_conformal_admet_shield():
    """Verify full ConformalADMETShield evaluation on structured ADMET profile."""
    shield = ConformalADMETShield(alpha=0.10)

    mock_profile = {
        "absorption": {
            "caco2_wang": {"value": -4.85, "unit": "log10(cm/s)"},
            "hia_hou": {"probability": 0.96},
            "bioavailability_ma": {"probability": 0.52},
        },
        "distribution": {
            "bbb_martins": {"probability": 0.08},
            "vdss_lombardo": {"value": 0.35, "unit": "log10(L/kg)"},
        },
        "toxicity": {
            "herg": {"probability": 0.04},
            "ames": {"probability": 0.95},
        },
    }

    evaluated = shield.evaluate_profile(mock_profile)

    assert "caco2_wang" in evaluated
    assert "lower_bound" in evaluated["caco2_wang"]
    assert "upper_bound" in evaluated["caco2_wang"]

    assert "hia_hou" in evaluated
    assert evaluated["hia_hou"]["prediction_set"] == [1]

    assert "bbb_martins" in evaluated
    assert evaluated["bbb_martins"]["prediction_set"] == [0]

    assert "bioavailability_ma" in evaluated
    # Probability 0.52 should yield [0, 1] uncertain set
    assert evaluated["bioavailability_ma"]["prediction_set"] == [0, 1]

    assert "herg" in evaluated
    assert evaluated["herg"]["prediction_set"] == [0]

    assert "ames" in evaluated
    assert evaluated["ames"]["prediction_set"] == [1]


def test_conformal_api_endpoints():
    """Verify conformal endpoints on FastAPI application."""
    from fastapi.testclient import TestClient

    from tdc_studio.serving.app import app

    client = TestClient(app)

    # 1. Test POST /predict/conformal
    resp = client.post(
        "/predict/conformal",
        json={"smiles": ["CC(=O)Oc1ccccc1C(=O)O"], "conformal_alpha": 0.10},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "results" in data
    assert len(data["results"]) == 1
    assert data["results"][0]["confidence_level"] == 0.90
    assert "uncertainty_by_task" in data["results"][0]

    # 2. Test POST /predict/admet_full with include_conformal=True
    resp_full = client.post(
        "/predict/admet_full",
        json={"smiles": ["CC(=O)Oc1ccccc1C(=O)O"], "include_conformal": True},
    )
    assert resp_full.status_code == 200, resp_full.text
    full_data = resp_full.json()
    prof = full_data["results"][0]
    assert prof["conformal_uncertainty"] is not None
    assert "caco2_wang" in prof["conformal_uncertainty"]
    assert "lower_bound" in prof["conformal_uncertainty"]["caco2_wang"]
