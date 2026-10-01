"""Synthesis route visualizer for Mermaid diagrams and ASCII terminal trees."""

import re

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

            lines.append(
                f'    {rxn_id}{{"Step {step.step_number}: {step.rule_name}\\n(Yield: {step.yield_pct:.1f}%)"}}:::reaction'
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
            f"📊 Solved: {route.solved} | Depth: {route.total_depth} steps | Cumulative Yield: {route.cumulative_yield}% | Cost: ${route.total_cost:.2f}",
            "=" * 70,
        ]

        stock_set = set(route.starting_materials)
        body = []

        for step in reversed(route.steps):
            body.append(f"Step {step.step_number}: [{step.rule_name}] -> Yield: {step.yield_pct:.1f}%")
            body.append(f"  └── Product: {step.product}")
            body.append("  └── Precursors:")
            for r in step.reactants:
                tag = " (📦 Stock BB)" if r in stock_set else " (🔄 Intermediate)"
                body.append(f"      ├── {r}{tag}")
            body.append("-" * 50)

        return "\n".join(header + body)
