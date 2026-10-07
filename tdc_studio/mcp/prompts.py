"""Prompt templates for TDC-Studio MCP Server."""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer


def register_prompts(server: MCPServer) -> None:
    """Register domain workflow prompts with the MCP server."""

    @server.prompt("lead_optimization_workflow", description="Closed-loop Lead Optimization Protocol")
    def lead_optimization_workflow(smiles: str, target_profile: str = "balanced") -> str:
        """Prompt instructing the agent to run closed-loop lead optimization."""
        return f"""You are acting as an expert Lead Optimization Medicinal Chemist.
Target Molecule SMILES: {smiles}
Optimization Priority: {target_profile}

Please execute the following 5-step closed-loop workflow:
1. Call `predict_admet_profile` to diagnose the parent molecule's core liabilities (hERG, AMES, Clearance, CYP, PPBR).
2. Call `explain_toxicity_hotspots` to pinpoint the specific atom/bond toxicophore substructures.
3. Call `optimize_lead_molecule` with bioisosteric transformations to generate structurally-conserved derivatives.
4. Call `simulate_pbpk_regimen` on the top candidate to verify that repeat-dose QD or BID reaches therapeutic steady-state while maintaining a >30x hERG safety margin.
5. Call `plan_retrosynthesis_route` to ensure the candidate has a commercially viable synthesis route (<= 5 steps from in-stock catalog reagents).
Synthesize your findings with clear medicinal chemistry rationales for why the recommended candidate is superior.
"""

    @server.prompt("ind_safety_review", description="ICH CTD Module 2.4/2.6 Nonclinical Safety & PK Review")
    def ind_safety_review(smiles: str, target_name: str = "Oncology Target", dose_mg: str = "100.0") -> str:
        """Prompt instructing the agent to review candidate safety and draft CTD narrative."""
        return f"""You are acting as a Senior Regulatory Affairs and Nonclinical Safety Scientist.
Candidate SMILES: {smiles}
Target Biological Mechanism: {target_name}
Planned Clinical Dose: {dose_mg} mg

Please conduct a formal nonclinical assessment:
1. Retrieve `guidelines://ich_ctd_ind` and `guidelines://fda_ddi` for regulatory context.
2. Call `predict_admet_profile` and `evaluate_drug_interactions` to identify any CYP perpetrator contraindications.
3. Call `simulate_pbpk_regimen` to calculate the in vivo Cmax, Css_avg, and hERG safety margin.
4. Call `compile_candidate_dossier` with your scientific rationale to generate the formal IND Dossier report.
Provide a clear Go / No-Go decision recommendation based on ICH M4 and FDA safety guidelines.
"""
