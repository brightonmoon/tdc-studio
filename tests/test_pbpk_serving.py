"""Tests for PBPK serving endpoint and PBPKServingPipeline."""

import pytest
from fastapi.testclient import TestClient

from tdc_studio.serving.app import app, set_pbpk_pipeline
from tdc_studio.serving.pbpk_pipeline import PBPKServingPipeline


class MockPPBRPipeline:
    def predict(self, smiles_list):
        return {"predictions": [80.0 for _ in smiles_list]}


class MockVDssPipeline:
    def predict(self, smiles_list):
        # 10^0.0 = 1.0 L/kg
        return {"predictions": [0.0 for _ in smiles_list]}


class MockClearancePipeline:
    def predict(self, smiles_list):
        return {
            "predictions": {
                "half_life_obach": [0.69897 for _ in smiles_list],  # 10^0.69897 ~ 5.0 hr
                "clearance_microsome_az": [1.477 for _ in smiles_list],  # 10^1.477 ~ 30 uL/min/mg
                "clearance_hepatocyte_az": [
                    1.0 for _ in smiles_list
                ],  # 10^1.0 = 10 uL/min/10^6 cells
            }
        }


def test_pbpk_serving_pipeline_calculation():
    pipe = PBPKServingPipeline(
        ppbr_pipeline=MockPPBRPipeline(),
        vdss_pipeline=MockVDssPipeline(),
        clearance_pipeline=MockClearancePipeline(),
    )
    results = pipe.predict_pbpk(["CC(=O)NC1=CC=C(O)C=C1"])
    assert len(results) == 1
    res = results[0]
    assert res["smiles"] == "CC(=O)NC1=CC=C(O)C=C1"
    assert res["ppbr_percent"] == 80.0
    assert res["unbound_fraction_fu"] == 0.20
    assert pytest.approx(res["vdss_l_kg"], rel=1e-2) == 1.0
    assert pytest.approx(res["half_life_hr"], rel=1e-2) == 5.0
    # CL = (1.0 * ln2) / 5.0 = 0.1386 L/hr/kg = 2.31 mL/min/kg
    assert pytest.approx(res["cl_total_l_hr_kg"], rel=1e-2) == 0.1386
    assert pytest.approx(res["cl_total_ml_min_kg"], rel=1e-2) == 2.3105
    assert res["extraction_class"] is not None


def test_pbpk_api_endpoint():
    pipe = PBPKServingPipeline(
        ppbr_pipeline=MockPPBRPipeline(),
        vdss_pipeline=MockVDssPipeline(),
        clearance_pipeline=MockClearancePipeline(),
    )
    set_pbpk_pipeline(pipe)

    client = TestClient(app)
    response = client.post(
        "/predict/pbpk",
        json={"smiles": ["CC(=O)NC1=CC=C(O)C=C1", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]},
    )
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) == 2
    assert data["model_name"] == "TDC-Studio-PBPK-Pipeline"
    assert "cl_total_ml_min_kg" in data["results"][0]
