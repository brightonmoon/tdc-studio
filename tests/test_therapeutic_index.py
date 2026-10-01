"""Unit and integration tests for Therapeutic Index and Clinical Developability Engine."""

import pytest

from tdc_studio.evaluation.therapeutic_index import (
    ComponentScores,
    TherapeuticIndexEngine,
    TherapeuticIndexProfile,
)


def test_herg_calibration():
    # p=0.5 corresponds to TDC benchmark threshold 10 uM (10,000 nM)
    ic50_mid = TherapeuticIndexEngine.calibrate_herg_ic50(0.50)
    assert pytest.approx(ic50_mid, rel=1e-2) == 10000.0

    # Low blocker prob -> high IC50 (safe)
    ic50_low = TherapeuticIndexEngine.calibrate_herg_ic50(0.10)
    assert ic50_low > 10000.0

    # High blocker prob -> low IC50 (toxic)
    ic50_high = TherapeuticIndexEngine.calibrate_herg_ic50(0.90)
    assert ic50_high < 10000.0


def test_potency_scoring():
    # Sub-nanomolar potency -> 25 pts
    assert TherapeuticIndexEngine.calculate_potency_score(0.5) == 25.0
    # 10 nM -> ~22 pts
    assert 21.0 <= TherapeuticIndexEngine.calculate_potency_score(10.0) <= 23.0
    # Weak potency (>1000 nM) -> low score
    assert TherapeuticIndexEngine.calculate_potency_score(5000.0) < 5.0


def test_safety_window_scoring():
    # >= 100x margin -> full 25 pts
    assert TherapeuticIndexEngine.calculate_safety_window_score(150.0) == 25.0
    # 30x ~ 50x -> moderate
    assert 14.0 <= TherapeuticIndexEngine.calculate_safety_window_score(40.0) <= 20.0
    # < 10x -> very low
    assert TherapeuticIndexEngine.calculate_safety_window_score(5.0) < 5.0


def test_organ_toxicology_scoring():
    # Clean molecule
    score_clean = TherapeuticIndexEngine.calculate_organ_toxicology_score(0.1, 0.1, 0.1)
    assert score_clean == 25.0

    # Positive AMES + DILI + ClinTox -> penalized heavily
    score_toxic = TherapeuticIndexEngine.calculate_organ_toxicology_score(0.9, 0.9, 0.9)
    assert score_toxic < 15.0


def test_pk_scoring():
    # Ideal QD half-life (6h) and low extraction ratio
    score_ideal = TherapeuticIndexEngine.calculate_pk_score(half_life_hr=6.0, extraction_ratio_eh=0.2)
    assert score_ideal >= 20.0

    # Rapid half-life (< 1h) and high hepatic clearance
    score_poor = TherapeuticIndexEngine.calculate_pk_score(half_life_hr=0.5, extraction_ratio_eh=0.85)
    assert score_poor < 15.0


def test_therapeutic_index_compute_aspirin():
    engine = TherapeuticIndexEngine(device="cpu")
    aspirin = "CC(=O)Oc1ccccc1C(=O)O"

    profile = engine.compute(
        smiles=aspirin,
        target_kd_nm=5.0,  # 5 nM target potency
        dose_mg=100.0,
    )

    assert isinstance(profile, TherapeuticIndexProfile)
    assert profile.smiles == aspirin
    assert profile.target_kd_nm == 5.0
    assert profile.target_pkd > 8.0
    assert profile.herg_ic50_nm > 0.0
    assert profile.herg_safety_margin > 0.0
    assert profile.herg_risk_tier in [
        "Safe (Margin >= 100x)",
        "Borderline (Caution 30x-100x)",
        "High Risk (<30x)",
    ]
    assert 0.0 <= profile.clinical_developability_score <= 100.0
    assert profile.developability_tier in [
        "Tier 1: High Clinical Potential",
        "Tier 2: Moderate Potential / Lead Optimization Required",
        "Tier 3: High Liability Risk",
    ]
    assert isinstance(profile.component_scores, ComponentScores)

    # Test serialization
    data = profile.to_dict()
    assert data["smiles"] == aspirin
    assert "clinical_developability_score" in data
    assert "component_scores" in data


def test_therapeutic_index_explicit_herg():
    engine = TherapeuticIndexEngine(device="cpu")
    # Low Kd (1 nM) and high explicit hERG (100,000 nM) -> 100,000x margin
    profile = engine.compute(
        smiles="c1ccccc1",
        target_kd_nm=1.0,
        herg_ic50_nm=100000.0,
    )
    assert profile.herg_safety_margin == 100000.0
    assert profile.herg_risk_tier == "Safe (Margin >= 100x)"
    assert profile.herg_therapeutic_window_log10 >= 4.0


def test_therapeutic_index_invalid_smiles():
    engine = TherapeuticIndexEngine(device="cpu")
    with pytest.raises(ValueError, match="Invalid SMILES"):
        engine.compute("INVALID_NOT_A_SMILES")
