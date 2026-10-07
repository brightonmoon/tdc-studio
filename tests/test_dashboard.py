"""Tests for the interactive 3-in-1 web dashboard."""

from fastapi.testclient import TestClient

from tdc_studio.serving.app import app

client = TestClient(app)


def test_dashboard_root_endpoint():
    """Verify GET / returns 200 OK with dashboard HTML content."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    html = response.text
    assert "TDC-Studio" in html
    assert "Next-Gen ADMET, DTI & Retrosynthesis Platform" in html
    assert "vpop-svg-chart" in html
    assert "retro-flowchart-container" in html
    assert "Aspirin" in html


def test_dashboard_named_endpoint():
    """Verify GET /dashboard returns 200 OK with the exact dashboard interface."""
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    html = response.text
    assert "TDC-Studio" in html
    assert "Retrosynthesis Search Config" in html
    assert "Physiologically Based Pharmacokinetics" in html
    assert "Aspirin" in html
