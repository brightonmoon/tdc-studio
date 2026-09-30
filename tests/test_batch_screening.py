"""Tests for High-Throughput Batch Molecular Screening Engine and API."""

import io
from fastapi.testclient import TestClient
from rdkit import Chem

from tdc_studio.serving.app import app
from tdc_studio.serving.batch_engine import (
    BatchScreeningEngine,
    compute_lipinski_rule_of_5,
    parse_molecular_file,
)
from tdc_studio.serving.unified_pipeline import UnifiedADMETPipeline


def test_compute_lipinski_rule_of_5():
    # Aspirin: MW ~ 180.16, LogP ~ 1.31, HBD = 1, HBA = 4, TPSA ~ 63.6
    mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
    ro5 = compute_lipinski_rule_of_5(mol)
    assert ro5["ro5_pass"] is True
    assert ro5["ro5_violations"] == 0
    assert 170.0 < ro5["mw"] < 190.0
    assert ro5["hbd"] == 1
    assert ro5["hba"] == 3


def test_parse_molecular_file_csv():
    csv_data = """id,Drug,target
CMPD_01,CC(=O)Oc1ccccc1C(=O)O,TargetA
CMPD_02,CC(C)Cc1ccc(cc1)C(C)C(=O)O,TargetB
"""
    records = parse_molecular_file(csv_data, "test_compounds.csv")
    assert len(records) == 2
    assert records[0]["compound_id"] == "CMPD_01"
    assert records[0]["smiles"] == "CC(=O)Oc1ccccc1C(=O)O"
    assert records[1]["compound_id"] == "CMPD_02"


def test_batch_screening_process():
    pipe = UnifiedADMETPipeline()
    engine = BatchScreeningEngine(pipeline=pipe)

    records = [
        {"compound_id": "Aspirin", "smiles": "CC(=O)Oc1ccccc1C(=O)O"},
        {"compound_id": "Ibuprofen", "smiles": "CC(C)Cc1ccc(cc1)C(C)C(=O)O"},
    ]

    df = engine.process_records(records)
    assert len(df) == 2
    assert "compound_id" in df.columns
    assert "mw" in df.columns
    assert "ro5_pass" in df.columns
    assert "c1_caco2_wang_val" in df.columns
    assert "c1_lipophilicity_astrazeneca_val" in df.columns
    assert "c2_ppbr_az_val" in df.columns
    assert "c4_clearance_hepatocyte_az_val" in df.columns
    assert "c5_herg_prob" in df.columns
    assert "c5_ames_prob" in df.columns
    assert "pbpk_cl_total_l_h_kg" in df.columns


def test_batch_api_endpoints():
    client = TestClient(app)
    csv_content = b"""id,smiles
Aspirin,CC(=O)Oc1ccccc1C(=O)O
Caffeine,CN1C=NC2=C1C(=O)N(C(=O)N2C)C
"""
    # 1. Test /predict/batch_preview
    resp_prev = client.post(
        "/predict/batch_preview",
        files={"file": ("test.csv", io.BytesIO(csv_content), "text/csv")},
    )
    assert resp_prev.status_code == 200
    data = resp_prev.json()
    assert "summary" in data
    assert data["summary"]["total_molecules"] == 2
    assert len(data["preview_data"]) == 2

    # 2. Test /predict/batch_file (CSV streaming download)
    resp_file = client.post(
        "/predict/batch_file?export_format=csv",
        files={"file": ("test.csv", io.BytesIO(csv_content), "text/csv")},
    )
    assert resp_file.status_code == 200
    assert "text/csv" in resp_file.headers["content-type"]
    assert "compound_id" in resp_file.text
    assert "Aspirin" in resp_file.text
    assert "Caffeine" in resp_file.text
