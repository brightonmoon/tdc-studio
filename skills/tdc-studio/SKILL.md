---
name: tdc-studio
description: Specialized Therapeutics AI Skill for molecular property prediction (25-Task ADMET), repeat-dose PBPK simulation, mechanism-based CYP drug-drug interactions (DDI), commercial catalog retrosynthesis pathway planning, and automated ICH CTD nonclinical IND candidate dossier generation.
---

# TDC-Studio Agent Skill: AI-Native Drug Discovery & Therapeutics Engine

This skill guides AI agents (Antigravity, Claude Code, Cursor, Devin) on when and how to utilize the `tdc-studio` Command-Line Interface (CLI) to design, evaluate, simulate, and document novel small molecule therapeutics.

---

## 🎯 When to Use This Skill

Activate this skill whenever the user asks to:
1. **Predict ADMET Properties**: Evaluate Absorption (Caco-2, HIA, LogD7.4, LogS), Distribution (PPBR, VDss, BBB), CYP450 Metabolism, Clearance/Half-Life, or Safety/Toxicity (hERG, AMES, DILI, ClinTox).
2. **Simulate Pharmacokinetics (PBPK)**: Calculate single-dose or repeat-dose (QD, BID) steady-state kinetics ($C_{ss,\max}, C_{ss,\min}, R_{ac}$, %PTF) and clinical cardiotoxicity margins ($IC_{50} / C_{\max} \ge 30\times$).
3. **Assess Drug-Drug Interactions (DDI)**: Quantify CYP-mediated inhibition and victim drug AUC fold changes on FDA standard probe substrates (Midazolam, Warfarin, Omeprazole, etc.).
4. **Plan Retrosynthesis Routes**: Search A* multi-step reaction pathways from target SMILES to commercially available catalog reagents (Enamine, Sigma).
5. **Optimize Lead Molecules**: Generate closed-loop bioisosteric analogues that alleviate toxicological liabilities while verifying retrosynthetic accessibility.
6. **Compile Regulatory Reports**: Generate comprehensive FDA IND / ICH CTD Module 2.4/2.6 nonclinical Candidate Evaluation Dossiers (HTML & JSON).

---

## 💻 CLI Command Quick Reference

All commands should be executed in the repository root using `uv run tdc-studio <command>`.

### 1. 25-Task ADMET & In Vivo PBPK Prediction
Predicts the full 25 ADMET indicators and primary pharmacokinetic parameters from SMILES:
```bash
uv run tdc-studio predict "<SMILES>"
```
*Output*: Formatted Rich console table displaying indicators, values, and decision tiers (Green/Amber/Red).

### 2. Clinical Developability & Therapeutic Index (TI)
Evaluates target affinity ($K_d$), in vivo free drug safety margin, and Clinical Developability Index (CDI, 0~100):
```bash
uv run tdc-studio ti "<SMILES>" --kd <KD_IN_NM> --dose <DOSE_IN_MG>
```
*Example*:
```bash
uv run tdc-studio ti "CC(=O)Oc1ccccc1C(=O)O" --kd 15.0 --dose 100.0
```

### 3. One-Click IND Candidate Dossier Generation
Compiles a complete 5-Chapter nonclinical dossier integrating ADMET, PBPK, CYP DDI, and Retro* into standalone HTML and machine-readable JSON:
```bash
uv run tdc-studio dossier "<SMILES>" --target <TARGET_GENE> --dose <DOSE_MG> --output-dir reports
```
*Key Flag*:
- `--output-dir <DIR>`: Specifies the output folder (defaults to `reports`). Generates `reports/dossier_<TARGET>.html` and `reports/dossier_<TARGET>.json`.

### 4. Multi-Step Retrosynthesis Planning (Retro*)
Searches commercially viable synthesis pathways from catalog building blocks:
```bash
uv run tdc-studio retrosynthesis plan --smiles "<SMILES>" --top-k 3 --max-depth 5 --render-mermaid --output route.json
```
*Key Flags*:
- `--top-k <INT>`: Number of diverse alternative routes to return (default: 3).
- `--banned "<SMILES1>,<SMILES2>"`: Blacklist specific unavailable or hazardous reagents.
- `--render-mermaid`: Outputs executable Mermaid.js flowchart syntax for diagrams.
- `--output <FILE>`: Saves detailed route steps and starting material catalog IDs to JSON.

### 5. Persistent In-Memory Daemon (MCP Server)
If your workflow involves multiple back-to-back queries (>5 calls), avoid repeated Python cold-starts by starting the MCP server in the background:
```bash
# In-process standard I/O (for direct MCP clients)
uv run tdc-studio mcp --transport stdio

# Local HTTP SSE server (for background REST/WebSocket agents)
uv run tdc-studio mcp --transport sse --host 127.0.0.1 --port 8000
```

---

## 🔬 Recommended Agent Workflow Patterns

### Workflow A: Closed-Loop Lead Optimization & Validation
When asked to optimize a molecule with liabilities:
1. **Diagnosis**: Run `uv run tdc-studio predict "<PARENT_SMILES>"` to identify liabilities (e.g. hERG $> 0.30$, AMES+, or $CL_{\text{int}} > 50$).
2. **Derivatization**: Identify toxicophore substructures and test bioisosteric replacements (e.g. basic amine $\to$ morpholine/ether).
3. **PBPK Verification**: For the best analogue, run `uv run tdc-studio ti "<ANALOGUE_SMILES>" --kd 10.0 --dose 50.0` to confirm that the hERG safety margin is $\ge 30\times$.
4. **Synthesis Feasibility**: Run `uv run tdc-studio retrosynthesis plan --smiles "<ANALOGUE_SMILES>" --top-k 1` to ensure it can be synthesized within $\le 4$ steps from catalog stock.
5. **Dossier Publication**: Run `uv run tdc-studio dossier "<ANALOGUE_SMILES>" --target <TARGET> --output-dir reports` to produce the final client deliverable.

### Workflow B: Clinical Drug-Drug Interaction (DDI) Safety Review
When asked about co-administration safety:
1. Run `uv run tdc-studio dossier "<SMILES>" --target <TARGET> --dose <DOSE_MG>`.
2. Inspect the resulting `dossier_<TARGET>.json` under `"cyp_ddi_evaluation"`.
3. Check for `"has_severe_ddi_risk": true` and read `"contraindicated_drugs"`.
4. Report whether sensitive index substrates (such as S-Warfarin or Midazolam) will experience an AUC Fold Change $\ge 2.0\times$ (moderate) or $\ge 5.0\times$ (strong), and formulate clinical dose adjustment recommendations.

---

## ⚠️ Best Practices & Guardrails for AI Agents

1. **Avoid Stdout Bloat**: For complex analyses (dossiers and multi-route retrosynthesis), always provide `--output` or `--output-dir`. Do not dump full 10,000-line JSON payloads into stdout; read only the specific summary keys required.
2. **Canonicalize SMILES**: Always ensure SMILES strings are chemically valid. If uncertain, test with RDKit before passing to CLI.
3. **Interpret in Context**: Never just report numbers. Always contextualize:
   - $CL_{\text{mic}} < 15 \mu\text{L/min/mg}$ means high metabolic stability.
   - $R_{ac} \approx 1.33$ means moderate accumulation compatible with once-daily (QD) oral dosing.
   - hERG safety margin $< 30\times$ requires medicinal chemistry structural modification.
