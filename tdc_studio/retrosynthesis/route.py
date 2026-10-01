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
    rank: int = 1
    rank_score: float = 0.0

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
            "rank": self.rank,
            "rank_score": self.rank_score,
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
            rank=data.get("rank", 1),
            rank_score=data.get("rank_score", 0.0),
        )
        return route


class RouteDiversityEvaluator:
    """Evaluates pairwise similarity and distance between two retrosynthesis routes."""

    @staticmethod
    def compute_similarity(route_a: RetrosynthesisRoute, route_b: RetrosynthesisRoute) -> float:
        """Calculate Jaccard similarity in [0.0, 1.0] based on starting materials and reaction steps."""
        # 1. Starting materials Jaccard
        sm_a = set(route_a.starting_materials)
        sm_b = set(route_b.starting_materials)
        if not sm_a and not sm_b:
            sm_sim = 1.0
        elif not sm_a or not sm_b:
            sm_sim = 0.0
        else:
            sm_sim = len(sm_a & sm_b) / len(sm_a | sm_b)

        # 2. Reaction steps signature Jaccard (rules and products)
        steps_a = set((s.rule_name, s.product) for s in route_a.steps)
        steps_b = set((s.rule_name, s.product) for s in route_b.steps)
        if not steps_a and not steps_b:
            step_sim = 1.0
        elif not steps_a or not steps_b:
            step_sim = 0.0
        else:
            step_sim = len(steps_a & steps_b) / len(steps_a | steps_b)

        return 0.6 * sm_sim + 0.4 * step_sim

    @classmethod
    def compute_distance(cls, route_a: RetrosynthesisRoute, route_b: RetrosynthesisRoute) -> float:
        """Distance = 1.0 - similarity."""
        return round(1.0 - cls.compute_similarity(route_a, route_b), 4)

    @classmethod
    def filter_diverse_routes(
        cls,
        routes: List[RetrosynthesisRoute],
        max_routes: int = 3,
        diversity_threshold: float = 0.25,
    ) -> List[RetrosynthesisRoute]:
        """Filter a list of solved routes to ensure diversity among candidates."""
        if not routes:
            return []

        diverse = [routes[0]]
        for candidate in routes[1:]:
            # Check distance against all already accepted routes
            too_similar = False
            for accepted in diverse:
                dist = cls.compute_distance(candidate, accepted)
                if dist < diversity_threshold:
                    too_similar = True
                    break
            if not too_similar:
                diverse.append(candidate)
            if len(diverse) >= max_routes:
                break

        # If not enough diverse routes were found, fill with remaining unique routes
        if len(diverse) < max_routes:
            for candidate in routes:
                if candidate not in diverse:
                    diverse.append(candidate)
                if len(diverse) >= max_routes:
                    break

        return diverse


class RouteRanker:
    """Ranks candidate routes using multi-objective cost, yield, and depth criteria."""

    @staticmethod
    def rank_routes(routes: List[RetrosynthesisRoute]) -> List[RetrosynthesisRoute]:
        """Score and sort routes by ascending rank_score (lower is better)."""
        if not routes:
            return []

        costs = [r.total_cost for r in routes]
        max_cost = max(costs) if costs and max(costs) > 0 else 1.0

        for r in routes:
            r.calculate_metrics()
            cost_norm = r.total_cost / max(1.0, max_cost)
            yield_penalty = max(0.0, 1.0 - (r.cumulative_yield / 100.0))
            depth_penalty = r.total_depth / 10.0
            r.rank_score = round(0.45 * cost_norm + 0.40 * yield_penalty + 0.15 * depth_penalty, 4)

        # Sort by rank_score ascending (solved routes first)
        sorted_routes = sorted(
            routes, key=lambda x: (not x.solved, x.rank_score, x.total_cost, -x.cumulative_yield)
        )
        for idx, r in enumerate(sorted_routes, start=1):
            r.rank = idx

        return sorted_routes

