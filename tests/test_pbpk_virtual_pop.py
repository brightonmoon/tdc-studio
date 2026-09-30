"""Tests for PBPK Virtual Population Monte Carlo simulation and API endpoints."""

import pytest
from fastapi.testclient import TestClient

from tdc_studio.pbpk.engine import HumanPhysiologicalParams, PBPKEngine, PBPKProfile
from tdc_studio.pbpk.virtual_population import (
    PopulationSubgroup,
    VirtualPopulationEngine,
    VirtualPopulationSimulationResult,
)
from tdc_studio.serving.app import app


@pytest.fixture
def baseline_aspirin_profile() -> PBPKProfile:
    """Fixture providing a mock/standard baseline PBPK profile for Aspirin."""
    engine = PBPKEngine()
    return engine.calculate_profile(
        smiles="CC(=O)Oc1ccccc1C(=O)O",
        vdss_l_kg=0.15,
        half_life_hr=0.35,
        ppbr_percent=85.0,
    )


def test_virtual_population_simulation_healthy(baseline_aspirin_profile):
    """Verify Monte Carlo simulation for healthy adult population."""
    engine = VirtualPopulationEngine(random_seed=42)
    n_subjects = 100
    res = engine.simulate(
        smiles="CC(=O)Oc1ccccc1C(=O)O",
        baseline_profile=baseline_aspirin_profile,
        subgroup=PopulationSubgroup.HEALTHY_ADULTS,
        n_subjects=n_subjects,
        dose_mg=100.0,
        t_max_sim_hours=24.0,
        n_timepoints=25,
    )

    assert isinstance(res, VirtualPopulationSimulationResult)
    assert res.n_subjects == n_subjects
    assert res.subgroup == "healthy_adults"
    assert res.dose_mg == 100.0

    # Statistical consistency checks
    assert res.vdss_l_kg.median > 0
    assert res.vdss_l_kg.p5 < res.vdss_l_kg.p95
    assert res.cl_total_l_h_kg.cv_pct > 0
    assert res.half_life_hours.p5 < res.half_life_hours.p95

    # Trajectory validation
    traj = res.trajectory
    assert len(traj.time_hours) == 25
    assert len(traj.median_ug_ml) == 25
    assert traj.time_hours[0] == 0.0
    assert traj.median_ug_ml[0] == 0.0  # At t=0 oral concentration is 0
    # Peak concentration should be reached within simulation
    assert max(traj.median_ug_ml) > 0.0


def test_virtual_population_renal_and_hepatic_impairment(baseline_aspirin_profile):
    """Verify physiological impairment effects: reduced clearance and increased exposure."""
    engine = VirtualPopulationEngine(random_seed=123)

    healthy_res = engine.simulate(
        smiles="CC(=O)Oc1ccccc1C(=O)O",
        baseline_profile=baseline_aspirin_profile,
        subgroup=PopulationSubgroup.HEALTHY_ADULTS,
        n_subjects=150,
    )

    renal_severe_res = engine.simulate(
        smiles="CC(=O)Oc1ccccc1C(=O)O",
        baseline_profile=baseline_aspirin_profile,
        subgroup=PopulationSubgroup.RENAL_SEVERE,
        n_subjects=150,
    )

    hepatic_severe_res = engine.simulate(
        smiles="CC(=O)Oc1ccccc1C(=O)O",
        baseline_profile=baseline_aspirin_profile,
        subgroup=PopulationSubgroup.HEPATIC_CHILD_PUGH_C,
        n_subjects=150,
    )

    # Renal severe should exhibit lower median clearance than healthy
    assert renal_severe_res.cl_total_l_h_kg.median < healthy_res.cl_total_l_h_kg.median
    # Hepatic Child-Pugh C should have higher median AUC (slower clearance)
    assert hepatic_severe_res.auc_inf_ug_h_ml.median > healthy_res.auc_inf_ug_h_ml.median
    # Hepatic Child-Pugh C should have prolonged half life
    assert hepatic_severe_res.half_life_hours.median > healthy_res.half_life_hours.median


def test_virtual_population_api_endpoint():
    """Verify POST /pbpk/virtual_population API endpoint."""
    client = TestClient(app)

    payload = {
        "smiles": "CC(=O)Oc1ccccc1C(=O)O",
        "subgroup": "geriatric",
        "n_subjects": 50,
        "dose_mg": 50.0,
        "ka_per_h": 1.5,
        "t_max_sim_hours": 12.0,
    }

    resp = client.post("/pbpk/virtual_population", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()

    assert data["subgroup"] == "geriatric"
    assert data["n_subjects"] == 50
    assert data["dose_mg"] == 50.0

    metrics = data["metrics"]
    assert "vdss_l_kg" in metrics
    assert "cl_total_l_h_kg" in metrics
    assert "half_life_hours" in metrics
    assert "auc_inf_ug_h_ml" in metrics
    assert metrics["half_life_hours"]["median"] > 0

    traj = data["trajectory"]
    assert "time_hours" in traj
    assert "median_ug_ml" in traj
    assert "p5_ug_ml" in traj
    assert "p95_ug_ml" in traj
