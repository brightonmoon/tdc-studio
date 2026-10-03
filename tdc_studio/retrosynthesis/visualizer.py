"""Synthesis route visualizer for Mermaid diagrams and ASCII terminal trees."""

import re
from typing import Any, Dict, List

from tdc_studio.retrosynthesis.route import RetrosynthesisRoute


def _clean_node_id(smiles: str, prefix: str = "N") -> str:
    """Generate safe, valid Mermaid node identifier from SMILES string."""
    cleaned = re.sub(r"[^a-zA-Z0-9]", "_", smiles)
    return f"{prefix}_{cleaned[:20]}_{abs(hash(smiles)) % 10000}"


class RouteVisualizer:
    """Visualizes RetrosynthesisRoute instances as Mermaid diagrams or ASCII trees."""

    @staticmethod
    def to_mermaid(route: RetrosynthesisRoute) -> str:
        """Render route into Mermaid flowchart syntax."""
        if not route.steps:
            target_id = _clean_node_id(route.target_smiles, "T")
            status = "In Stock" if route.solved else "No Route Found"
            return (
                "```mermaid\n"
                "flowchart TD\n"
                f'    {target_id}["Target: {route.target_smiles}\\n({status})"]\n'
                "```"
            )

        lines = [
            "```mermaid",
            "flowchart TD",
            "    %% Styles",
            "    classDef target fill:#ff7675,stroke:#d63031,stroke-width:2px,color:#fff;",
            "    classDef stock fill:#55efc4,stroke:#00b894,stroke-width:2px,color:#2d3436;",
            "    classDef intermediate fill:#74b9ff,stroke:#0984e3,stroke-width:1px,color:#fff;",
            "    classDef reaction fill:#ffeaa7,stroke:#fdcb6e,stroke-width:1px,color:#2d3436;",
        ]

        target_id = _clean_node_id(route.target_smiles, "T")
        lines.append(f'    {target_id}["🎯 Target\\n{route.target_smiles}"]:::target')

        stock_set = set(route.starting_materials)
        node_map = {route.target_smiles: target_id}

        for i, step in enumerate(route.steps, start=1):
            rxn_id = f"RXN_{i}"
            prod_id = node_map.get(step.product, _clean_node_id(step.product, f"P{i}"))
            node_map[step.product] = prod_id

            cond_info = ""
            if step.conditions:
                cat = step.conditions.get("catalyst")
                reag = step.conditions.get("reagents", [])
                temp = step.conditions.get("temperature_c")
                parts = []
                if cat:
                    parts.append(f"Cat: {cat}")
                elif reag:
                    parts.append(f"Reag: {reag[0]}")
                if temp is not None:
                    parts.append(f"{temp:.0f}°C")
                if parts:
                    cond_info = f"\\n[{', '.join(parts)}]"

            lines.append(
                f'    {rxn_id}{{"Step {step.step_number}: {step.rule_name}\\n(Yield: {step.yield_pct:.1f}%){cond_info}"}}:::reaction'
            )
            lines.append(f"    {rxn_id} --> {prod_id}")

            for r_idx, reactant in enumerate(step.reactants):
                if reactant not in node_map:
                    r_id = _clean_node_id(reactant, f"R{i}_{r_idx}")
                    node_map[reactant] = r_id
                    if reactant in stock_set:
                        lines.append(f'    {r_id}["📦 Stock BB\\n{reactant}"]:::stock')
                    else:
                        lines.append(f'    {r_id}["Intermediate\\n{reactant}"]:::intermediate')
                else:
                    r_id = node_map[reactant]

                lines.append(f"    {r_id} --> {rxn_id}")

        lines.append("```")
        return "\n".join(lines)

    @staticmethod
    def to_text_tree(route: RetrosynthesisRoute) -> str:
        """Render route into readable ASCII tree format."""
        if not route.steps:
            return f"Target: {route.target_smiles} (Solved: {route.solved}, Depth: 0)"

        header = [
            "=" * 70,
            f"🎯 Target Molecule: {route.target_smiles}",
            f"📊 Solved: {route.solved} | Depth: {route.total_depth} steps | Cumulative Yield: {route.cumulative_yield}% | Total Cost: ${route.total_cost:.2f}/g | SCS: {route.synthetic_complexity_score:.1f}/10",
            "=" * 70,
        ]

        stock_set = set(route.starting_materials)
        body = []

        for step in reversed(route.steps):
            body.append(
                f"Step {step.step_number}: [{step.rule_name}] -> Yield: {step.yield_pct:.1f}% | Cost: ${step.cost:.2f}/g"
            )
            if step.conditions and step.conditions.get("summary"):
                body.append(f"  ├── Protocol: {step.conditions['summary']}")
            if step.cost_breakdown:
                cb = step.cost_breakdown
                body.append(
                    f"  ├── Breakdown: Raw: ${cb.get('materials_cost', 0):.1f} | Ops: ${cb.get('operational_cost', 0):.1f} | Purif: ${cb.get('purification_cost', 0):.1f} | Risk: ${cb.get('risk_penalty', 0):.1f}"
                )
            body.append(f"  └── Product: {step.product}")
            body.append("  └── Precursors:")
            for r in step.reactants:
                tag = " (📦 Stock BB)" if r in stock_set else " (🔄 Intermediate)"
                body.append(f"      ├── {r}{tag}")
            body.append("-" * 50)

        return "\n".join(header + body)

    @staticmethod
    def to_comparison_summary(routes: List[RetrosynthesisRoute]) -> List[Dict[str, Any]]:
        """Extract structured comparison summary for multiple candidate routes."""
        summary = []
        for r in routes:
            r.calculate_metrics()
            rxn_rules = list(dict.fromkeys(s.rule_name for s in r.steps))
            summary.append(
                {
                    "rank": r.rank,
                    "solved": r.solved,
                    "total_depth": r.total_depth,
                    "cumulative_yield": r.cumulative_yield,
                    "total_cost": r.total_cost,
                    "tcs_cost": r.tcs_cost,
                    "synthetic_complexity_score": r.synthetic_complexity_score,
                    "starting_materials_count": len(r.starting_materials),
                    "starting_materials": r.starting_materials,
                    "reaction_rules": rxn_rules,
                    "rank_score": r.rank_score,
                }
            )
        return summary

    @staticmethod
    def to_comparison_table(routes: List[RetrosynthesisRoute]) -> str:
        """Render a comparative Markdown table across all candidate routes."""
        if not routes:
            return "No retrosynthesis routes available."

        header = [
            "| 순위 (Rank) | 상태 (Status) | 단계 수 (Depth) | 누적 수율 (Yield) | 예상 비용 (TCS) | 난이도 (SCS) | 출발 물질 (Starting BBs) | 주요 반응 (Reactions) |",
            "| :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |",
        ]

        rank_badges = {1: "🥇 1위 (최적)", 2: "🥈 2위 (대안 A)", 3: "🥉 3위 (대안 B)", 4: "4위 (대안 C)", 5: "5위 (대안 D)"}

        rows = []
        for r in routes:
            badge = rank_badges.get(r.rank, f"{r.rank}위")
            status = "✅ Solved" if r.solved else "❌ Unsolved"
            depth = f"{r.total_depth}단계"
            cum_yield = f"{r.cumulative_yield:.1f}%"
            cost = f"${r.total_cost:.2f}/g"
            scs = f"{r.synthetic_complexity_score:.1f}/10"
            bb_str = ", ".join(r.starting_materials[:3])
            if len(r.starting_materials) > 3:
                bb_str += f" 외 {len(r.starting_materials)-3}종"
            rules = ", ".join(dict.fromkeys(s.rule_name for s in r.steps)) or "N/A"

            rows.append(
                f"| {badge} | {status} | {depth} | {cum_yield} | {cost} | {scs} | `{bb_str}` | {rules} |"
            )

        return "\n".join(header + rows)

    @classmethod
    def to_multi_route_mermaid(cls, routes: List[RetrosynthesisRoute]) -> str:
        """Render individual Mermaid flowcharts for each candidate route with rank headers."""
        if not routes:
            return ""

        sections = []
        rank_titles = {
            1: "### 🥇 1순위 최적 경로 (Champion Route)",
            2: "### 🥈 2순위 대안 경로 (Alternative Route A)",
            3: "### 🥉 3순위 대안 경로 (Alternative Route B)",
            4: "### 4순위 대안 경로 (Alternative Route C)",
            5: "### 5순위 대안 경로 (Alternative Route D)",
        }

        for r in routes:
            title = rank_titles.get(r.rank, f"### {r.rank}순위 대안 경로")
            summary_info = f"**단계 수:** {r.total_depth} | **누적 수율:** {r.cumulative_yield:.1f}% | **비용:** ${r.total_cost:.2f}"
            mermaid = cls.to_mermaid(r)
            sections.append(f"{title}\n{summary_info}\n\n{mermaid}")

        return "\n\n---\n\n".join(sections)

