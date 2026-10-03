"""Unit tests for Candidate Evaluation Dossier module."""

import json
import os

from tdc_studio.dossier.collector import DossierCollector
from tdc_studio.dossier.models import DossierDataPayload
from tdc_studio.dossier.renderer import DossierRenderer


def test_dossier_collector_basic():
    """Test data collection for a standard candidate compound (e.g. Aspirin)."""
    collector = DossierCollector()
    smiles = "CC(=O)Oc1ccccc1C(=O)O"

    payload = collector.collect(
        smiles=smiles,
        target_name="COX-1",
        target_kd_nm=15.0,
        dose_mg=100.0,
    )

    assert isinstance(payload, DossierDataPayload)
    assert payload.candidate.canonical_smiles == "CC(=O)Oc1ccccc1C(=O)O"
    assert payload.candidate.molecular_weight > 170.0
    assert "<svg" in payload.candidate.structure_svg

    # Check filters
    assert payload.filter_results["is_valid"] is True
    assert payload.filter_results["pains_passed"] is True

    # Check synthesizability / retrosynthesis
    assert payload.retrosynthesis is not None
    assert "passed" in payload.retrosynthesis


def test_dossier_collector_with_target_sequence():
    """Test data collection including target protein residue attention."""
    collector = DossierCollector()
    smiles = "c1ccccc1"
    target_seq = "ACDEFGHIKLMNPQRSTVWY"

    payload = collector.collect(
        smiles=smiles,
        target_seq=target_seq,
        target_name="TestTarget",
    )

    assert payload.target_attention is not None
    assert len(payload.target_attention["top_hotspot_residues"]) > 0


def test_dossier_renderer_html_and_json(tmp_path):
    """Test HTML and JSON rendering and file export."""
    collector = DossierCollector()
    smiles = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"  # Ibuprofen

    payload = collector.collect(smiles=smiles, target_name="COX-2")

    # Render HTML
    html_content = DossierRenderer.render_html(payload)
    assert "<!DOCTYPE html>" in html_content
    assert "Candidate Evaluation Dossier" in html_content
    assert payload.candidate.canonical_smiles in html_content

    # Render JSON
    json_str = DossierRenderer.render_json(payload)
    parsed = json.loads(json_str)
    assert parsed["candidate"]["canonical_smiles"] == payload.candidate.canonical_smiles
    assert "filters" in parsed

    # Export to disk
    html_path = str(tmp_path / "report.html")
    exported_path = DossierRenderer.export_file(payload, html_path, output_format="html")
    assert os.path.exists(exported_path)
    with open(exported_path, "r", encoding="utf-8") as f:
        read_content = f.read()
    assert "<!DOCTYPE html>" in read_content
