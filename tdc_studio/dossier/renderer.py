"""Dossier Renderer: Generates Standalone Interactive HTML and JSON Evaluation Sheets.

Outputs zero-dependency, self-contained HTML reports featuring embedded SVG structures,
CDI scorecard meters, and toxicity safety margin summaries for decision-making.
"""

import html
import json
import logging
import os

from tdc_studio.dossier.models import DossierDataPayload

logger = logging.getLogger("tdc_studio.dossier.renderer")


class DossierRenderer:
    """Renders candidate dossier payload into presentation formats."""

    @classmethod
    def render_json(cls, payload: DossierDataPayload, indent: int = 2) -> str:
        """Render machine-readable formatted JSON."""
        return json.dumps(payload.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def render_html(cls, payload: DossierDataPayload) -> str:
        """Render self-contained HTML evaluation sheet with inline CSS and SVGs."""
        cand = payload.candidate
        filters = payload.filter_results
        ti = payload.therapeutic_index or {}
        pbpk = payload.pbpk_simulation or {}
        retro = payload.retrosynthesis or {}
        attn = payload.target_attention or {}

        # CDI Scorecard
        cdi_score = ti.get("clinical_developability_score", 0.0)
        cdi_tier = ti.get("developability_tier", "Unassessed")
        tier_color = (
            "#10b981"
            if "Tier 1" in cdi_tier
            else ("#f59e0b" if "Tier 2" in cdi_tier else "#ef4444")
        )

        # Pillar scores
        pillars = ti.get("component_scores", {})
        potency_pt = pillars.get("potency", 0.0)
        safety_pt = pillars.get("safety_window", 0.0)
        organ_pt = pillars.get("organ_toxicology", 0.0)
        pk_pt = pillars.get("human_pk", 0.0)

        # Filters status
        pains_ok = filters.get("pains_passed", True)
        brenk_ok = filters.get("brenk_passed", True)
        ro5_ok = filters.get("ro5_passed", True)
        veber_ok = filters.get("veber_passed", True)

        # Retrosynthesis summary
        retro_passed = retro.get("passed", False)
        retro_steps = retro.get("total_depth", 1 if retro.get("tier2_1step_passed") else "-")
        retro_yield = retro.get(
            "cumulative_yield", "85.0" if retro.get("tier2_1step_passed") else "-"
        )
        starting_mats = retro.get(
            "starting_materials",
            ["Catalog Stock Precursor(s)"] if retro.get("tier2_1step_passed") else [],
        )

        # Target attention SVG
        attn_svg = ""
        if attn and "top_hotspot_residues" in attn:
            hotspots_list = attn["top_hotspot_residues"]
            hotspots_txt = ", ".join(
                [f"{h['label']} (score {h['score']})" for h in hotspots_list[:6]]
            )
            attn_svg = f"""
            <div class="card">
                <h3>🎯 Target Protein Binding Hotspots (Cross-Attention XAI)</h3>
                <p class="subtext">Identified key amino acid residues dominating non-covalent ligand interaction:</p>
                <div class="badges-row">
                    {"".join([f'<span class="badge badge-indigo">{h["label"]}</span>' for h in hotspots_list[:8]])}
                </div>
                <p style="font-size:0.85rem; color:#4b5563; margin-top:8px;">{html.escape(hotspots_txt)}</p>
            </div>
            """

        html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Candidate Evaluation Dossier - {html.escape(cand.canonical_smiles[:25])}...</title>
    <style>
        :root {{
            --primary: #2563eb;
            --primary-dark: #1d4ed8;
            --bg: #f8fafc;
            --card-bg: #ffffff;
            --text-main: #0f172a;
            --text-muted: #64748b;
            --border: #e2e8f0;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg);
            color: var(--text-main);
            padding: 32px 16px;
            line-height: 1.5;
        }}
        .container {{
            max-width: 960px;
            margin: 0 auto;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 2px solid var(--border);
            padding-bottom: 16px;
            margin-bottom: 24px;
        }}
        .header h1 {{ font-size: 1.75rem; font-weight: 800; color: #1e293b; }}
        .header .meta {{ font-size: 0.85rem; color: var(--text-muted); text-align: right; }}
        .grid-2 {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }}
        @media (max-width: 768px) {{
            .grid-2 {{ grid-template-columns: 1fr; }}
        }}
        .card {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            margin-bottom: 20px;
        }}
        .card h3 {{
            font-size: 1.1rem;
            font-weight: 700;
            color: #1e293b;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .subtext {{ font-size: 0.85rem; color: var(--text-muted); margin-bottom: 12px; }}
        .score-hero {{
            display: flex;
            align-items: center;
            gap: 24px;
        }}
        .score-circle {{
            width: 100px;
            height: 100px;
            border-radius: 50%;
            background: conic-gradient({tier_color} {cdi_score * 3.6}deg, #e2e8f0 0deg);
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
        }}
        .score-circle::after {{
            content: '{cdi_score:.0f}';
            width: 82px;
            height: 82px;
            border-radius: 50%;
            background: #fff;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.8rem;
            font-weight: 800;
            color: {tier_color};
        }}
        .pillar-bar {{
            margin-bottom: 8px;
        }}
        .pillar-label {{
            display: flex;
            justify-content: space-between;
            font-size: 0.8rem;
            font-weight: 600;
            color: #475569;
            margin-bottom: 3px;
        }}
        .bar-bg {{
            height: 8px;
            background: #e2e8f0;
            border-radius: 4px;
            overflow: hidden;
        }}
        .bar-fill {{
            height: 100%;
            border-radius: 4px;
            background: var(--primary);
        }}
        .table-data {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.88rem;
        }}
        .table-data th, .table-data td {{
            padding: 8px 12px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}
        .table-data th {{
            background: #f1f5f9;
            font-weight: 600;
            color: #475569;
        }}
        .badge {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 700;
        }}
        .badge-green {{ background: #dcfce7; color: #15803d; }}
        .badge-red {{ background: #fee2e2; color: #b91c1c; }}
        .badge-yellow {{ background: #fef3c7; color: #b45309; }}
        .badge-indigo {{ background: #e0e7ff; color: #4338ca; }}
        .badges-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin-top: 8px;
        }}
        .structure-container {{
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 220px;
            background: #fafafa;
            border-radius: 8px;
            padding: 12px;
        }}
        .structure-container svg {{
            max-width: 100%;
            height: auto;
            max-height: 240px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <div>
                <h1>🧬 Candidate Evaluation Dossier</h1>
                <p style="font-size:0.9rem; color:var(--text-muted); font-family:monospace; margin-top:4px;">
                    {html.escape(cand.canonical_smiles)}
                </p>
            </div>
            <div class="meta">
                <div><strong>TDC-Studio</strong> Report</div>
                <div>{payload.created_at}</div>
            </div>
        </div>

        <!-- 2D Structure & Physicochemical Profile -->
        <div class="grid-2">
            <div class="card">
                <h3>🔬 2D Chemical Structure</h3>
                <div class="structure-container">
                    {cand.structure_svg or "<p>No SVG Structure Available</p>"}
                </div>
            </div>

            <div class="card">
                <h3>📊 Physicochemical Druggability</h3>
                <table class="table-data">
                    <tr><th>Molecular Weight</th><td><strong>{cand.molecular_weight:.1f}</strong> g/mol</td></tr>
                    <tr><th>LogP (Lipophilicity)</th><td><strong>{cand.logp:.2f}</strong></td></tr>
                    <tr><th>TPSA (Polar Surface Area)</th><td><strong>{cand.tpsa:.1f}</strong> Å²</td></tr>
                    <tr><th>H-Bond Donors / Acceptors</th><td>{cand.hbd} / {cand.hba}</td></tr>
                    <tr><th>Rotatable Bonds</th><td>{cand.rotatable_bonds}</td></tr>
                    <tr><th>SAScore (Synthesizability)</th><td><strong>{cand.sa_score:.2f}</strong> (1=easy, 10=hard)</td></tr>
                </table>
                <div class="badges-row">
                    <span class="badge {"badge-green" if ro5_ok else "badge-red"}">Ro5: {"PASS" if ro5_ok else "FAIL"}</span>
                    <span class="badge {"badge-green" if veber_ok else "badge-red"}">Veber: {"PASS" if veber_ok else "FAIL"}</span>
                    <span class="badge {"badge-green" if pains_ok else "badge-red"}">PAINS: {"CLEAN" if pains_ok else "ALERT"}</span>
                    <span class="badge {"badge-green" if brenk_ok else "badge-red"}">Brenk: {"CLEAN" if brenk_ok else "ALERT"}</span>
                </div>
            </div>
        </div>

        <!-- Clinical Developability Index (CDI) Scorecard -->
        <div class="card">
            <h3>🏆 Clinical Developability Index (CDI Scorecard)</h3>
            <div class="grid-2" style="align-items:center;">
                <div class="score-hero">
                    <div class="score-circle"></div>
                    <div>
                        <div style="font-size:1.4rem; font-weight:800; color:{tier_color};">{cdi_tier}</div>
                        <p class="subtext">Integrated score out of 100 points reflecting potency, safety window, organ liabilities, and PK.</p>
                    </div>
                </div>
                <div>
                    <div class="pillar-bar">
                        <div class="pillar-label"><span>Potency (Target Kd)</span><span>{potency_pt:.1f} / 25</span></div>
                        <div class="bar-bg"><div class="bar-fill" style="width:{potency_pt * 4}%;"></div></div>
                    </div>
                    <div class="pillar-bar">
                        <div class="pillar-label"><span>Safety Window (hERG Margin)</span><span>{safety_pt:.1f} / 25</span></div>
                        <div class="bar-bg"><div class="bar-fill" style="width:{safety_pt * 4}%;"></div></div>
                    </div>
                    <div class="pillar-bar">
                        <div class="pillar-label"><span>Organ Toxicology (DILI/AMES)</span><span>{organ_pt:.1f} / 25</span></div>
                        <div class="bar-bg"><div class="bar-fill" style="width:{organ_pt * 4}%;"></div></div>
                    </div>
                    <div class="pillar-bar">
                        <div class="pillar-label"><span>Human Pharmacokinetics (PBPK)</span><span>{pk_pt:.1f} / 25</span></div>
                        <div class="bar-bg"><div class="bar-fill" style="width:{pk_pt * 4}%;"></div></div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Safety, PBPK & Retrosynthesis Highlights -->
        <div class="grid-2">
            <div class="card">
                <h3>🛡️ Safety & Exposure Margins</h3>
                <table class="table-data">
                    <tr><th>Target Potency (Kd)</th><td><strong>{ti.get("target_kd_nm", "-")}</strong> nM</td></tr>
                    <tr><th>hERG Inhibition IC50</th><td><strong>{ti.get("herg_ic50_nm", "-")}</strong> nM</td></tr>
                    <tr><th>In Vitro hERG Margin</th><td><strong>{ti.get("herg_safety_margin", "-")}x</strong></td></tr>
                    <tr><th>Predicted Cmax (free)</th><td><strong>{pbpk.get("cmax_free_ug_ml", "-")}</strong> ug/mL</td></tr>
                    <tr><th>In Vivo Exposure Margin</th><td><strong>{pbpk.get("in_vivo_herg_margin", "-")}x</strong></td></tr>
                </table>
            </div>

            <div class="card">
                <h3>🌳 Retro* Synthesizability & Route</h3>
                <table class="table-data">
                    <tr><th>Synthetic Status</th><td><span class="badge {"badge-green" if retro_passed else "badge-red"}">{"TRACTABLE" if retro_passed else "RESTRICTED"}</span></td></tr>
                    <tr><th>Route Depth</th><td><strong>{retro_steps}</strong> step(s)</td></tr>
                    <tr><th>Estimated Yield</th><td><strong>{retro_yield}%</strong></td></tr>
                    <tr><th>Starting Materials</th><td>{len(starting_mats)} stock compound(s)</td></tr>
                </table>
                <div class="badges-row">
                    {"".join([f'<span class="badge badge-indigo">{html.escape(s[:30])}</span>' for s in starting_mats[:3]])}
                </div>
            </div>
        </div>

        <!-- Target Attention Hotspots (if available) -->
        {attn_svg}

    </div>
</body>
</html>
"""
        return html_template

    @classmethod
    def export_file(
        cls,
        payload: DossierDataPayload,
        output_path: str,
        output_format: str = "html",
    ) -> str:
        """Export dossier report to HTML or JSON file on disk."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        if output_format.lower() == "json":
            content = cls.render_json(payload)
        else:
            content = cls.render_html(payload)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        return os.path.abspath(output_path)
