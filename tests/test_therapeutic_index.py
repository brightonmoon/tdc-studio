"""Comprehensive unit and integration tests for DTI-linked Therapeutic Index Pipeline."""

import pytest
from fastapi.testclient import TestClient

from tdc_studio.evaluation.therapeutic_index import (
    calculate_clinical_progression_score,
    calculate_dili_penalty,
    calculate_therapeutic_index,
    calibrate_herg_ic50_from_probability,
)
from tdc_studio.models.dti.dta_model import GraphDTAModel
from tdc_studio.serving.app import (
    app,
    set_dti_multi_pipeline,
    set_dti_pipeline,
    set_ti_pipeline,
    set_unified_pipeline,
)
from tdc_studio.serving.pipeline import DTIMultiAffinityPipeline
from tdc_studio.serving.schema import (
    HealthResponse,
    TherapeuticIndexResponse,
)
from tdc_studio.serving.therapeutic_index_pipeline import (
    TherapeuticIndexPipeline,
    set_therapeutic_index_pipeline,
)
from tdc_studio.serving.unified_pipeline import UnifiedADMETPipeline

# ==============================================================================
# 1. Pure Mathematical & Biophysical Unit Tests
# ==============================================================================


def test_therapeutic_index_math():
    """Verify TI logarithmic formula and threshold tier classifications."""
    # Case 1: Wide safety window (TI >= 2.0, >= 100x margin)
    # Kd = 10 nM, hERG IC50 = 10,000 nM -> ratio = 1000 -> log10 = 3.0
    ti_safe, tier_safe = calculate_therapeutic_index(kd_nm=10.0, herg_ic50_nm=10000.0)
    assert ti_safe == 3.0
    assert tier_safe == "Safe (Wide Window)"

    # Case 2: Boundary for Safe (TI == 2.0, 100x margin)
    ti_bnd, tier_bnd = calculate_therapeutic_index(kd_nm=50.0, herg_ic50_nm=5000.0)
    assert ti_bnd == 2.0
    assert tier_bnd == "Safe (Wide Window)"

    # Case 3: Moderate Risk (1.0 <= TI < 2.0, 10 ~ 100x margin)
    # Kd = 20 nM, hERG IC50 = 1,000 nM -> ratio = 50 -> log10 = 1.699
    ti_mod, tier_mod = calculate_therapeutic_index(kd_nm=20.0, herg_ic50_nm=1000.0)
    assert 1.0 <= ti_mod < 2.0
    assert tier_mod == "Moderate Risk"

    # Case 4: Critical Hazard (TI < 1.0, < 10x margin)
    # Kd = 200 nM, hERG IC50 = 100 nM -> ratio = 0.5 -> log10 = -0.301
    ti_crit, tier_crit = calculate_therapeutic_index(kd_nm=200.0, herg_ic50_nm=100.0)
    assert ti_crit < 1.0
    assert tier_crit == "Critical Hazard"

    # Case 5: Zero / negative edge-case protection (no division by zero)
    ti_zero, tier_zero = calculate_therapeutic_index(kd_nm=0.0, herg_ic50_nm=0.0)
    assert isinstance(ti_zero, float)
    assert tier_zero in ("Safe (Wide Window)", "Moderate Risk", "Critical Hazard")


def test_herg_ic50_calibration():
    """Verify pharmacological log-logistic conversion from probability to nM."""
    # 50% blocker probability at TDC 10 uM cutoff -> exactly 10,000 nM
    ic50_mid, dec_mid = calibrate_herg_ic50_from_probability(0.5, cutoff_nm=10000.0)
    assert ic50_mid == 10000.0
    assert dec_mid == "Moderate Risk"

    # High blocker probability (P = 0.90) -> ~1,111.11 nM
    ic50_high, dec_high = calibrate_herg_ic50_from_probability(0.9, cutoff_nm=10000.0)
    assert round(ic50_high, 0) == 1111.0
    assert dec_high == "High Risk (hERG Blocker)"

    # Low blocker probability (P = 0.10) -> 90,000 nM
    ic50_low, dec_low = calibrate_herg_ic50_from_probability(0.1, cutoff_nm=10000.0)
    assert ic50_low == 90000.0
    assert dec_low == "Low Cardiotoxicity Risk"

    # Boundary extremes [0.0, 1.0] -> must be clamped, no math domain error
    ic50_min, _ = calibrate_herg_ic50_from_probability(1.0)
    assert ic50_min >= 1.0

    ic50_max, _ = calibrate_herg_ic50_from_probability(0.0)
    assert ic50_max <= 10_000_000.0


def test_dili_penalty_calculation():
    """Verify graduated hepatotoxicity penalty curve."""
    # Low risk (P < 0.3) -> 0 penalty
    pen_low, dec_low = calculate_dili_penalty(0.15)
    assert pen_low == 0.0
    assert dec_low == "Low Hepatotoxicity Risk"

    # Moderate risk (P = 0.40) -> 15 * (0.1 / 0.2) = 7.5 pts
    pen_mod, dec_mod = calculate_dili_penalty(0.40)
    assert pen_mod == 7.5
    assert dec_mod == "Moderate Hepatotoxicity Risk"

    # High risk (P = 0.80) -> 15 + 25 * (0.3 / 0.5) = 30.0 pts
    pen_high, dec_high = calculate_dili_penalty(0.80)
    assert pen_high == 30.0
    assert dec_high == "Hepatotoxicity Risk (DILI+)"


def test_clinical_progression_scoring():
    """Verify integrated multi-objective clinical candidate scoring (0 to 100)."""
    # Ideal drug candidate: Kd = 5 nM, hERG IC50 = 50,000 nM (TI = 4.0), DILI = 0.1, DL = 90
    ideal = calculate_clinical_progression_score(
        kd_nm=5.0,
        herg_ic50_nm=50000.0,
        dili_prob=0.1,
        drug_likeness_score=90.0,
        herg_prob=0.1,
    )
    assert ideal["clinical_progression_score"] >= 80.0
    assert ideal["clinical_decision"] == "Recommended (Pass)"
    assert ideal["ti_tier"] == "Safe (Wide Window)"
    assert ideal["safety_radar"]["therapeutic_window"] == 100.0

    # Risky candidate: Kd = 200 nM, hERG IC50 = 400 nM (TI = 0.30), DILI = 0.85
    risky = calculate_clinical_progression_score(
        kd_nm=200.0,
        herg_ic50_nm=400.0,
        dili_prob=0.85,
        drug_likeness_score=50.0,
        herg_prob=0.9,
    )
    assert risky["clinical_progression_score"] < 50.0
    assert risky["clinical_decision"] == "Rejected (Critical Hazard)"
    assert risky["ti_tier"] == "Critical Hazard"

    # Moderate candidate
    mod = calculate_clinical_progression_score(
        kd_nm=30.0,
        herg_ic50_nm=1500.0,  # TI ~ 1.70
        dili_prob=0.25,
        drug_likeness_score=75.0,
        herg_prob=0.4,
    )
    assert 50.0 <= mod["clinical_progression_score"] <= 75.0
    assert mod["clinical_decision"] in ("Recommended (Pass)", "Caution (Moderate Risk)")


# ==============================================================================
# 2. Pipeline Orchestrator Unit Tests
# ==============================================================================


@pytest.fixture
def ti_pipeline_fixture():
    """Create a standalone TherapeuticIndexPipeline with CPU unified ADMET."""
    admet_p = UnifiedADMETPipeline(device="cpu")
    pipe = TherapeuticIndexPipeline(dti_pipeline=None, admet_pipeline=admet_p, device="cpu")
    set_therapeutic_index_pipeline(pipe)
    set_unified_pipeline(admet_p)
    yield pipe
    set_therapeutic_index_pipeline(None)
    set_unified_pipeline(None)


def test_ti_pipeline_evaluate_single_and_batch(ti_pipeline_fixture):
    """Verify TherapeuticIndexPipeline evaluation with zero-training mock setup."""
    smiles = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
    target_seq = "MSHHWGYGKHNGPEHWHKDFPIAKGERQ"

    item = ti_pipeline_fixture.evaluate_pair(
        smiles=smiles,
        target_seq=target_seq,
        herg_source="admet",
        include_admet_details=True,
    )

    assert item.smiles == smiles
    assert item.target_sequence == target_seq
    assert item.kd_nm > 0.0
    assert item.herg_ic50_nm > 0.0
    assert item.ti_tier in ("Safe (Wide Window)", "Moderate Risk", "Critical Hazard")
    assert item.clinical_decision in (
        "Recommended (Pass)",
        "Caution (Moderate Risk)",
        "Rejected (Critical Hazard)",
    )
    assert 0.0 <= item.clinical_progression_score <= 100.0
    assert item.elapsed_ms > 0.0
    assert item.admet_profile is not None
    assert "toxicity" in item.admet_profile

    # Batch evaluation
    batch_items = ti_pipeline_fixture.evaluate_batch(
        smiles_list=[smiles, "CC(=O)NC1=CC=C(O)C=C1"],
        target_sequences=[target_seq, target_seq],
    )
    assert len(batch_items) == 2


# ==============================================================================
# 3. FastAPI Serving Integration Tests
# ==============================================================================


def test_therapeutic_index_endpoint_fastapi(ti_pipeline_fixture):
    """Verify POST /predict/therapeutic-index end-to-end HTTP serving."""
    client = TestClient(app)

    # 1. Health check verification
    health_resp = client.get("/healthz")
    assert health_resp.status_code == 200
    health_data = HealthResponse(**health_resp.json())
    assert health_data.status == "healthy"
    assert health_data.therapeutic_index_ready is True

    # 2. Main prediction request
    payload = {
        "smiles": [
            "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
            "Cn1cnc2c1c(=O)n(c(=O)n2C)C",  # Caffeine
        ],
        "target_sequences": [
            "MSHHWGYGKHNGPEHWHKDFPIAKGERQ",
            "MDSSTGPGNSTEAELLALHRR",
        ],
        "herg_source": "admet",
        "include_admet_details": False,
    }

    resp = client.post("/predict/therapeutic-index", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    parsed = TherapeuticIndexResponse(**data)
    assert parsed.count == 2
    assert parsed.pipeline_version == "TDC-Studio-TI-v1"
    assert parsed.elapsed_ms > 0.0
    assert len(parsed.results) == 2

    first = parsed.results[0]
    assert first.smiles == payload["smiles"][0]
    assert first.kd_nm > 0.0
    assert first.herg_ic50_nm > 0.0
    assert first.ti_tier in ("Safe (Wide Window)", "Moderate Risk", "Critical Hazard")
    assert first.clinical_decision in (
        "Recommended (Pass)",
        "Caution (Moderate Risk)",
        "Rejected (Critical Hazard)",
    )
    assert "therapeutic_window" in first.safety_radar
    assert "cardiac_safety" in first.safety_radar


def test_therapeutic_index_validation_errors():
    """Verify input length mismatch and empty payload rejection."""
    client = TestClient(app)

    # Mismatch between SMILES and target sequences lengths
    bad_payload = {
        "smiles": ["CC(=O)Oc1ccccc1C(=O)O", "CC"],
        "target_sequences": ["MSHHWGYGKHNGPEHWHKDFPIAKGERQ"],
    }
    resp = client.post("/predict/therapeutic-index", json=bad_payload)
    assert resp.status_code == 422

    # Empty payload
    empty_payload = {
        "smiles": [],
        "target_sequences": [],
    }
    resp_empty = client.post("/predict/therapeutic-index", json=empty_payload)
    assert resp_empty.status_code == 422


def test_dti_model_pipeline_full_integration():
    """Verify TherapeuticIndexPipeline when explicitly bound with GraphDTA multi-affinity model."""
    multi_model_cfg = {
        "type": "graph_dta",
        "out_dim": 3,
        "drug_encoder": {"type": "gine", "in_dim": 14, "hidden_dim": 16, "num_layers": 1},
        "target_encoder": {"type": "protein_cnn", "out_dim": 16, "kernel_sizes": [3]},
        "fusion": {"type": "cross_attention", "hidden_dim": 16, "num_heads": 2, "out_dim": 3},
    }
    model = GraphDTAModel(multi_model_cfg)
    scaler_meta = {
        "task_stats": {
            0: {"name": "Kd", "mean": 7.2, "std": 2.8},
            1: {"name": "Ki", "mean": 7.0, "std": 2.9},
            2: {"name": "IC50", "mean": 6.8, "std": 3.0},
        },
        "log_transform": True,
    }
    dti_pipe = DTIMultiAffinityPipeline(model=model, device="cpu", scaler_meta=scaler_meta)
    set_dti_multi_pipeline(dti_pipe, meta=multi_model_cfg)
    set_dti_pipeline(dti_pipe, meta=multi_model_cfg)

    # Create TI pipeline using this DTI model
    admet_p = UnifiedADMETPipeline(device="cpu")
    ti_pipe = TherapeuticIndexPipeline(dti_pipeline=dti_pipe, admet_pipeline=admet_p, device="cpu")
    set_ti_pipeline(ti_pipe)

    client = TestClient(app)
    payload = {
        "smiles": ["CC(=O)Oc1ccccc1C(=O)O"],
        "target_sequences": ["MSHHWGYGKHNGPEHWHKDFPIAKGERQ"],
        "herg_source": "dti",
    }
    resp = client.post("/predict/therapeutic-index", json=payload)
    assert resp.status_code == 200
    res_data = resp.json()["results"][0]
    assert res_data["kd_nm"] > 0.0
    assert res_data["herg_ic50_nm"] > 0.0

    # Cleanup
    set_dti_pipeline(None)
    set_dti_multi_pipeline(None)
    set_ti_pipeline(None)
    set_unified_pipeline(None)
