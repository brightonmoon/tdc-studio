"""Unit and integration tests for FastAPI Retrosynthesis endpoints."""

import pytest
from fastapi.testclient import TestClient

from tdc_studio.serving.app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_retrosynthesis_single_step_endpoint(client):
    payload = {
        "smiles": "c1ccc(C(=O)NC)cc1",
        "top_k": 5,
    }
    response = client.post("/retrosynthesis/single-step", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["product_smiles"] == "c1ccc(C(=O)NC)cc1"
    assert data["count"] > 0
    assert len(data["candidates"]) > 0

    first_cand = data["candidates"][0]
    assert "reactants" in first_cand
    assert "confidence" in first_cand
    assert first_cand["confidence"] > 0.0


def test_retrosynthesis_plan_endpoint(client):
    payload = {
        "smiles": "c1ccc(C(=O)NC)cc1",
        "max_depth": 3,
        "timeout_sec": 3.0,
        "render_mermaid": True,
    }
    response = client.post("/retrosynthesis/plan", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert (
        data["target_smiles"] == "CNC(=O)c1ccccc1" or data["target_smiles"] == "c1ccc(C(=O)NC)cc1"
    )
    assert data["solved"] is True
    assert data["total_depth"] >= 1
    assert data["cumulative_yield"] > 0.0
    assert len(data["steps"]) >= 1

    # Check Mermaid diagram rendering
    assert data["mermaid_diagram"] is not None
    assert "flowchart TD" in data["mermaid_diagram"]
    assert "Target" in data["mermaid_diagram"]
