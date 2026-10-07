"""Unit and integration tests for TDC-Studio MCP Server."""

import json
import os

import pytest

from tdc_studio.mcp.server import create_mcp_server


@pytest.mark.anyio
async def test_mcp_server_metadata_and_registrations():
    server = create_mcp_server()
    assert server.name == "tdc-studio"
    assert "Therapeutics Data Commons" in server.instructions

    tools = await server.list_tools()
    tool_names = [t.name for t in tools]
    expected_tools = [
        "predict_admet_profile",
        "explain_toxicity_hotspots",
        "evaluate_target_affinity",
        "simulate_pbpk_regimen",
        "evaluate_drug_interactions",
        "plan_retrosynthesis_route",
        "optimize_lead_molecule",
        "compile_candidate_dossier",
    ]
    for exp in expected_tools:
        assert exp in tool_names, f"Expected tool {exp} missing from MCP server"

    resources = await server.list_resources()
    resource_uris = [r.uri for r in resources]
    assert "guidelines://ich_ctd_ind" in resource_uris
    assert "guidelines://fda_ddi" in resource_uris
    assert "admet://reference_ranges" in resource_uris

    prompts = await server.list_prompts()
    prompt_names = [p.name for p in prompts]
    assert "lead_optimization_workflow" in prompt_names
    assert "ind_safety_review" in prompt_names


@pytest.mark.anyio
async def test_mcp_admet_and_xai_tool_calls():
    server = create_mcp_server()

    # 1. ADMET Profile
    res_admet = await server.call_tool("predict_admet_profile", {"smiles": "CC(=O)Oc1ccccc1C(=O)O"})
    assert res_admet is not None
    admet_text = res_admet.content[0].text
    data = json.loads(admet_text)
    assert "smiles" in data
    assert "drug_likeness_radar_score" in data
    assert "liabilities_detected" in data

    # 2. Toxicity Hotspots & Bioisosteres
    res_xai = await server.call_tool(
        "explain_toxicity_hotspots", {"smiles": "CC(=O)Oc1ccccc1C(=O)O", "liability_task": "herg"}
    )
    xai_data = json.loads(res_xai.content[0].text)
    assert "recommended_bioisosteres" in xai_data
    assert len(xai_data["recommended_bioisosteres"]) > 0


@pytest.mark.anyio
async def test_mcp_pbpk_and_ddi_tool_calls():
    server = create_mcp_server()

    # 1. Repeat-Dose PBPK
    res_pbpk = await server.call_tool(
        "simulate_pbpk_regimen",
        {"smiles": "CC(=O)Oc1ccccc1C(=O)O", "dose_mg": 100.0, "interval_hr": 24.0},
    )
    pbpk_data = json.loads(res_pbpk.content[0].text)
    assert "steady_state_outcome" in pbpk_data
    assert pbpk_data["steady_state_outcome"]["accumulation_ratio_rac"] > 1.0
    assert "safety_assessment" in pbpk_data

    # 2. CYP DDI Evaluation
    res_ddi = await server.call_tool(
        "evaluate_drug_interactions", {"smiles": "CC(=O)Oc1ccccc1C(=O)O", "dose_mg": 100.0}
    )
    ddi_data = json.loads(res_ddi.content[0].text)
    assert "cyp_evaluations" in ddi_data
    assert len(ddi_data["cyp_evaluations"]) == 5


@pytest.mark.anyio
async def test_mcp_retro_and_optimizer_tool_calls():
    server = create_mcp_server()

    # 1. Retrosynthesis planning
    res_retro = await server.call_tool(
        "plan_retrosynthesis_route", {"smiles": "CC(=O)Nc1ccc(O)cc1", "max_depth": 3, "top_k": 1}
    )
    retro_data = json.loads(res_retro.content[0].text)
    assert "route_solved" in retro_data

    # 2. Lead Optimization
    res_opt = await server.call_tool(
        "optimize_lead_molecule", {"smiles": "CC(=O)Oc1ccccc1C(=O)O", "max_candidates": 2}
    )
    opt_data = json.loads(res_opt.content[0].text)
    assert "top_candidates" in opt_data


@pytest.mark.anyio
async def test_mcp_dossier_compilation_tool(tmp_path):
    server = create_mcp_server()

    out_dir = str(tmp_path / "reports")
    res_dossier = await server.call_tool(
        "compile_candidate_dossier",
        {
            "smiles": "CC(=O)Oc1ccccc1C(=O)O",
            "target_name": "COX2",
            "dose_mg": 100.0,
            "clinical_rationale": "Favorable anti-inflammatory candidate with robust safety profile.",
            "output_dir": out_dir,
        },
    )
    dossier_data = json.loads(res_dossier.content[0].text)
    assert "clinical_developability_index_cdi" in dossier_data
    assert "regulatory_verdict" in dossier_data
    assert "artifacts_generated" in dossier_data

    html_file = dossier_data["artifacts_generated"]["interactive_html_report"]
    assert os.path.exists(html_file)
    with open(html_file, "r", encoding="utf-8") as f:
        content = f.read()
        assert (
            "Nonclinical Review &amp; Medicinal Chemistry Rationale" in content
            or "Nonclinical Review" in content
        )
