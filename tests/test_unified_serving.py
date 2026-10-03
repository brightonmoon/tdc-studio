"""End-to-End integration tests for Unified 22 ADMET + PBPK Serving Pipeline."""

import pytest
from fastapi.testclient import TestClient

from tdc_studio.serving.app import app, set_unified_pipeline
from tdc_studio.serving.schema import (
    ExplainResponse,
    OptimizeResponse,
    UnifiedADMETResponse,
)
from tdc_studio.serving.unified_pipeline import UnifiedADMETPipeline


@pytest.fixture
def unified_pipeline():
    """Create a lightweight UnifiedADMETPipeline instance for fast zero-training testing."""
    pipe = UnifiedADMETPipeline(device="cpu")
    set_unified_pipeline(pipe)
    yield pipe
    set_unified_pipeline(None)


def test_unified_pipeline_predict_single(unified_pipeline):
    # Aspirin
    smiles = "CC(=O)Oc1ccccc1C(=O)O"
    profile = unified_pipeline.predict_single(smiles)

    assert profile.smiles == smiles
    assert profile.canonical_smiles is not None
    assert profile.elapsed_ms > 0
    assert 0.0 <= profile.drug_likeness_score <= 100.0

    # 1. Absorption (6 tasks)
    assert len(profile.absorption) == 6
    assert "caco2_wang" in profile.absorption
    assert "lipophilicity_astrazeneca" in profile.absorption
    assert "solubility_aqsoldb" in profile.absorption
    assert "hia_hou" in profile.absorption
    assert "bioavailability_ma" in profile.absorption
    assert "pgp_broccatelli" in profile.absorption

    # 2. Distribution (3 tasks)
    assert len(profile.distribution) == 3
    assert "ppbr_az" in profile.distribution
    assert "vdss_lombardo" in profile.distribution
    assert "bbb_martins" in profile.distribution

    # 3. Metabolism (8 tasks)
    assert len(profile.metabolism) == 8
    assert "cyp3a4_veith" in profile.metabolism
    assert "cyp2d6_veith" in profile.metabolism
    assert "cyp2c9_substrate" in profile.metabolism

    # 4. Excretion (3 tasks)
    assert len(profile.excretion) == 3
    assert "clearance_microsome_az" in profile.excretion
    assert "clearance_hepatocyte_az" in profile.excretion
    assert "half_life_obach" in profile.excretion

    # 5. Toxicity (5 tasks)
    assert len(profile.toxicity) >= 5
    assert "herg" in profile.toxicity
    assert "ames" in profile.toxicity
    assert "dili" in profile.toxicity
    assert "clintox" in profile.toxicity
    assert "ld50_zhu" in profile.toxicity

    # 6. PBPK Profile
    assert profile.pbpk is not None
    assert profile.pbpk.vdss_l_kg > 0.0
    assert profile.pbpk.half_life_hours > 0.0
    assert 0.0 < profile.pbpk.fraction_unbound <= 1.0
    assert profile.pbpk.cl_total_l_h_kg > 0.0


def test_unified_serving_endpoint_fastapi(unified_pipeline):
    client = TestClient(app)

    # 1. Health check
    health_resp = client.get("/healthz")
    assert health_resp.status_code == 200
    data = health_resp.json()
    assert data["status"] == "healthy"
    assert data["unified_ready"] is True

    # 2. Full ADMET prediction
    payload = {
        "smiles": [
            "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
            "Cn1cnc2c1c(=O)n(c(=O)n2C)C",  # Caffeine
        ]
    }
    resp = client.post("/predict/admet_full", json=payload)
    assert resp.status_code == 200

    parsed = UnifiedADMETResponse(**resp.json())
    assert len(parsed.results) == 2
    assert parsed.model_version == "TDC-Studio-Unified-v1"

    # Verify first compound
    asp = parsed.results[0]
    assert asp.smiles == "CC(=O)Oc1ccccc1C(=O)O"
    assert "caco2_wang" in asp.absorption
    assert "ppbr_az" in asp.distribution
    assert "cyp3a4_veith" in asp.metabolism
    assert "clearance_hepatocyte_az" in asp.excretion
    assert "herg" in asp.toxicity
    assert asp.pbpk is not None


def test_unified_serving_invalid_payload():
    client = TestClient(app)
    # Empty smiles list should fail validation
    resp = client.post("/predict/admet_full", json={"smiles": []})
    assert resp.status_code == 422


def test_explain_endpoint_fastapi():
    client = TestClient(app)
    payload = {
        "smiles": "O=[N+]([O-])c1ccccc1",  # Nitrobenzene (known mutagen/AMES alert)
        "liability_focus": "ames",
        "steps": 10,
    }
    resp = client.post("/explain", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    parsed = ExplainResponse(**data)

    assert parsed.canonical_smiles == "O=[N+]([O-])c1ccccc1"
    assert parsed.num_atoms == 9
    assert len(parsed.atom_attributions) == 9
    assert len(parsed.normalized_attributions) == 9
    assert parsed.svg_data_uri.startswith("data:image/svg+xml;base64,")
    # Nitro group bioisosteres should be recommended
    assert len(parsed.bioisostere_recommendations) > 0
    nitro_swaps = [r.modified_smiles for r in parsed.bioisostere_recommendations]
    assert any("N#C" in s or "C#N" in s or "C(=O)N" in s for s in nitro_swaps)


def test_optimize_endpoint_fastapi():
    client = TestClient(app)
    payload = {
        "smiles": "O=[N+]([O-])c1ccccc1",  # Nitrobenzene
        "target_liability": "ames",
        "max_candidates": 3,
        "sa_threshold": 4.0,
    }
    resp = client.post("/optimize", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    parsed = OptimizeResponse(**data)

    assert parsed.canonical_smiles == "O=[N+]([O-])c1ccccc1"
    assert parsed.bemis_murcko_scaffold == "c1ccccc1"
    assert parsed.candidates_generated > 0
    assert parsed.candidates_passing_sa_filter > 0
    assert len(parsed.top_candidates) > 0

    top = parsed.top_candidates[0]
    assert top.sa_score <= 4.0
    assert top.scaffold_preserved is True
    assert top.transformation_name in [
        "nitro_to_cyano",
        "nitro_to_trifluoromethyl",
        "nitro_to_primary_amide",
    ]
