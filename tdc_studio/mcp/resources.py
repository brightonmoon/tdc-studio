"""Domain Knowledge Resources for TDC-Studio MCP Server."""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer


def register_resources(server: MCPServer) -> None:
    """Register all domain knowledge resources with the MCP server."""

    @server.resource("guidelines://ich_ctd_ind", name="ICH CTD IND Regulatory Guidance")
    def get_ich_ctd_guidance() -> str:
        """ICH M4 CTD Nonclinical Structure Guidance for Investigational New Drug (IND) Applications."""
        return """# ICH M4 Common Technical Document (CTD) - Nonclinical Overview (Module 2.4 & 2.6)

## 1. Module 2.4: Nonclinical Overview
- **2.4.1 General Aspects**: Rationale for the drug development program, biological target validation, and therapeutic indication.
- **2.4.2 Content and Structural Format**: Overview of primary pharmacodynamics, secondary pharmacodynamics, safety pharmacology, pharmacokinetics, and toxicology.
- **2.4.3 Pharmacodynamics (PD)**: In vitro target binding affinity (Kd, Ki, IC50), target selectivity, mechanism of action, and efficacy proof-of-concept.
- **2.4.4 Pharmacokinetics (PK / ADME)**:
  - Absorption: Caco-2 permeability (P_app), Bioavailability (F), HIA.
  - Distribution: Plasma protein binding (%PPB, fu), Volume of distribution (Vd,ss), Blood-Brain Barrier (BBB).
  - Metabolism: Metabolic stability, CYP450 5-major isoform inhibition and substrate profiling (1A2, 2C9, 2C19, 2D6, 3A4).
  - Excretion: Systemic clearance (CL_total, CL_hepatic), elimination half-life (t_1/2), IVIVE extrapolation.
  - Drug-Drug Interactions (DDI): AUC fold change on standard victim substrates (e.g. Midazolam, Warfarin).
- **2.4.5 Toxicology (Safety)**:
  - Safety Pharmacology: Cardiotoxicity (hERG potassium channel inhibition IC50 and clinical safety margin >= 30x).
  - Genetic Toxicology: AMES mutagenicity bacterial reverse mutation test, Ashby-Tennant structural alerts.
  - Target Organ Toxicity: Drug-Induced Liver Injury (DILI), ClinTox FDA clinical failure defense.
  - Therapeutic Index (TI) & Clinical Developability Index (CDI): Margin of safety between therapeutic exposure and toxicity thresholds.
- **2.4.6 Integrated Risk Assessment & Justification of First-in-Human (FIH) Dose**: Allometric scaling and Maximum Recommended Starting Dose (MRSD).
"""

    @server.resource("guidelines://fda_ddi", name="FDA Clinical Drug-Drug Interaction Criteria")
    def get_fda_ddi_guidance() -> str:
        """FDA Clinical Drug-Drug Interaction (DDI) Evaluation Criteria and Standard Index Substrates."""
        return """# FDA Clinical Drug-Drug Interaction (DDI) Guidance Summary

## 1. Classification of In Vivo CYP Inhibitors
Based on AUC Fold Change of sensitive index victim substrates:
- **Strong Inhibitor**: >= 5.0-fold increase in victim drug AUC (or >= 80% decrease in clearance).
  *Clinical Impact*: Co-administration contraindicated or requires severe dosage adjustments.
- **Moderate Inhibitor**: >= 2.0-fold to < 5.0-fold increase in victim drug AUC (or 50-80% decrease in clearance).
  *Clinical Impact*: Caution advised; dose reduction and close clinical/therapeutic drug monitoring required.
- **Weak Inhibitor**: >= 1.25-fold to < 2.0-fold increase in victim drug AUC (or 20-50% decrease in clearance).
  *Clinical Impact*: Mild interaction; general clinical vigilance.
- **No Clinically Relevant Interaction**: < 1.25-fold increase in victim drug AUC.

## 2. Standard Sensitive FDA Probe Substrates
- **CYP3A4**: Midazolam (fm = 0.92, highly sensitive index substrate)
- **CYP2C9**: S-Warfarin (fm = 0.85, Narrow Therapeutic Index [NTI] anticoagulant - critical bleeding risk)
- **CYP2C19**: Omeprazole (fm = 0.82)
- **CYP2D6**: Dextromethorphan (fm = 0.90)
- **CYP1A2**: Caffeine (fm = 0.88)
"""

    @server.resource("admet://reference_ranges", name="TDC ADMET 25-Task Reference Ranges")
    def get_admet_reference_ranges() -> str:
        """Standard Physiological Reference Ranges and Decision Tiers for 25 ADMET Endpoints."""
        return """# TDC ADMET 25-Task Reference Ranges & Quality Criteria

## Absorption (C1)
- **Caco-2 Permeability**: High (> -5.15 log cm/s), Moderate (-6.0 ~ -5.15), Low (< -6.0)
- **Lipophilicity (LogD7.4)**: Optimal (1.0 ~ 3.0), Hydrophilic (< 1.0), Lipophilic Liability (> 3.0)
- **Aqueous Solubility (LogS)**: Soluble (> -2.0 log mol/L), Moderate (-4.0 ~ -2.0), Poorly Soluble (< -4.0)
- **Human Intestinal Absorption (HIA)**: Favorable (Prob >= 0.50), Poor (Prob < 0.50)
- **Bioavailability (F)**: Bioavailable (Prob >= 0.50), Low (Prob < 0.50)

## Distribution (C2)
- **Plasma Protein Binding (PPBR)**: Low (<= 80%), Moderate (80% ~ 95%), High Binding (> 95%)
- **Volume of Distribution (VDss)**: Low (< 0.7 L/kg), Moderate (0.7 ~ 2.0 L/kg), High (> 2.0 L/kg)
- **Blood-Brain Barrier (BBB)**: Penetrant (Prob >= 0.50), Non-penetrant (Prob < 0.50)

## Metabolism (C3)
- **CYP 5-Isoforms (1A2, 2C9, 2C19, 2D6, 3A4)**: Non-inhibitor (Prob < 0.50), Inhibitor (Prob >= 0.50)
- **Substrate Turnover**: Rapid Clearance (Prob >= 0.50), Stable (Prob < 0.50)

## Excretion & PBPK (C4)
- **Microsomal Clearance**: Low (< 15 uL/min/mg), Moderate (15 ~ 50), High Turnover (> 50)
- **Hepatocyte Clearance**: Low (< 10 uL/min/10^6 cells), Moderate (10 ~ 30), High (> 30)
- **Elimination Half-Life (t1/2)**: Short (< 2h), Moderate (2h ~ 8h), Long (> 8h, QD compatible)

## Safety & Toxicity (C5)
- **hERG Cardiotoxicity**: Safe (Prob < 0.30), Moderate Risk (0.30 ~ 0.70), High Liability (Prob > 0.70)
- **AMES Mutagenicity**: Safe Negative (Prob < 0.50), Mutagenic Positive (Prob >= 0.50)
- **Drug-Induced Liver Injury (DILI)**: Safe (Prob < 0.50), Hepatotoxic (Prob >= 0.50)
- **ClinTox (FDA Phase Failures)**: Approved (Prob < 0.50), Toxic Failure (Prob >= 0.50)
"""
