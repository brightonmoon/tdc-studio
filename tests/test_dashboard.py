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
    assert "3-in-1 Cockpit" in html
    assert "admetRadarChart" in html
    assert "pbpkLineChart" in html
    assert "svgContainer" in html
    assert "pymolScript" in html
    assert "candidateTableBody" in html


def test_dashboard_named_endpoint():
    """Verify GET /dashboard returns 200 OK with the exact dashboard interface."""
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    html = response.text
    assert "Self-Correcting Generative Lead Optimizer" in html
    assert "Terfenadine" in html
    assert "Aspirin" in html
