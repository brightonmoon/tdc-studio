"""Data structures for retrosynthetic pathways, reaction steps, and route trees."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class ReactionStep:
    """Represents a single chemical reaction transformation step."""

    step_number: int
    reactants: List[str]
    product: str
    rule_name: str
    confidence: float
    yield_pct: float = 80.0
    cost: float = 10.0


@dataclass
class RetrosynthesisRoute:
    """Complete multi-step synthetic pathway from target to commercial building blocks."""

    target_smiles: str
    steps: List[ReactionStep] = field(default_factory=list)
    solved: bool = False
    total_depth: int = 0
    cumulative_yield: float = 100.0
    total_cost: float = 0.0
    starting_materials: List[str] = field(default_factory=list)

    def calculate_metrics(self) -> None:
        """Recompute depth, cumulative yield, and total cost across all steps."""
        self.total_depth = len(self.steps)
        if not self.steps:
            self.cumulative_yield = 100.0
            self.total_cost = 0.0
            return

        cum_y = 1.0
        tot_c = 0.0
        all_reactants = set()
        produced = set()

        for s in self.steps:
            cum_y *= (s.yield_pct / 100.0)
            tot_c += s.cost
            produced.add(s.product)
            for r in s.reactants:
                all_reactants.add(r)

        self.cumulative_yield = round(cum_y * 100.0, 2)
        self.total_cost = round(tot_c, 2)
        # Starting materials are reactants that were not synthesized in any previous step
        self.starting_materials = sorted(list(all_reactants - produced))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize route to standard JSON-compatible dictionary."""
        return {
            "target_smiles": self.target_smiles,
            "solved": self.solved,
            "total_depth": self.total_depth,
            "cumulative_yield": self.cumulative_yield,
            "total_cost": self.total_cost,
            "starting_materials": self.starting_materials,
            "steps": [asdict(s) for s in self.steps],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetrosynthesisRoute":
        """Deserialize from dictionary."""
        steps = [ReactionStep(**s) for s in data.get("steps", [])]
        route = cls(
            target_smiles=data["target_smiles"],
            steps=steps,
            solved=data.get("solved", False),
            total_depth=data.get("total_depth", len(steps)),
            cumulative_yield=data.get("cumulative_yield", 100.0),
            total_cost=data.get("total_cost", 0.0),
            starting_materials=data.get("starting_materials", []),
        )
        return route
