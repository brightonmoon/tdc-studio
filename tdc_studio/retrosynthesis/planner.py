"""End-to-End Retrosynthesis Planning Orchestrator."""

import logging
from typing import Optional

from tdc_studio.models.retrosynthesis.forward_verifier import ForwardVerifier
from tdc_studio.models.retrosynthesis.hybrid_policy import HybridRetroPolicy
from tdc_studio.models.retrosynthesis.rule_policy import RuleRetroPolicy
from tdc_studio.models.retrosynthesis.yield_predictor import YieldPredictor
from tdc_studio.retrosynthesis.route import RetrosynthesisRoute
from tdc_studio.retrosynthesis.search.retro_star import RetroStarSearcher
from tdc_studio.retrosynthesis.stock import StockLibrary
from tdc_studio.retrosynthesis.visualizer import RouteVisualizer

logger = logging.getLogger("tdc_studio.retrosynthesis.planner")


class RetroPlanner:
    """High-level Retrosynthesis Planning Engine."""

    def __init__(
        self,
        policy_type: str = "rule",
        max_depth: int = 5,
        beam_width: int = 5,
        timeout_sec: float = 5.0,
        verify_round_trip: bool = False,
    ):
        self.stock = StockLibrary(load_builtin=True)
        self.verifier = ForwardVerifier()
        self.yield_predictor = YieldPredictor({"hidden_dim": 64, "n_bits": 512})

        if policy_type == "hybrid":
            self.policy = HybridRetroPolicy()
        else:
            self.policy = RuleRetroPolicy()

        self.searcher = RetroStarSearcher(
            policy=self.policy,
            stock=self.stock,
            verifier=self.verifier,
            yield_predictor=self.yield_predictor,
            max_depth=max_depth,
            beam_width=beam_width,
            timeout_sec=timeout_sec,
            verify_round_trip=verify_round_trip,
        )

    def plan_routes(
        self,
        target_smiles: str,
        top_k: int = 3,
        diversity_threshold: float = 0.25,
        banned_smiles: Optional[list] = None,
        max_depth: Optional[int] = None,
        timeout_sec: Optional[float] = None,
    ) -> list:
        """Find top-k distinct synthetic pathways for target molecule."""
        if max_depth is not None:
            self.searcher.max_depth = max_depth
        return self.searcher.search_top_k(
            target_smiles=target_smiles,
            top_k=top_k,
            diversity_threshold=diversity_threshold,
            banned_smiles=banned_smiles,
            timeout_sec=timeout_sec,
        )

    def plan_route(
        self,
        target_smiles: str,
        max_depth: Optional[int] = None,
        timeout_sec: Optional[float] = None,
    ) -> RetrosynthesisRoute:
        """Find complete single optimal synthetic pathway for target molecule."""
        routes = self.plan_routes(
            target_smiles=target_smiles,
            top_k=1,
            max_depth=max_depth,
            timeout_sec=timeout_sec,
        )
        return routes[0]

    def is_route_found(
        self,
        target_smiles: str,
        max_depth: int = 4,
        timeout_sec: float = 2.0,
    ) -> bool:
        """Fast check if target has a valid synthetic route to commercial stock."""
        route = self.plan_route(target_smiles, max_depth=max_depth, timeout_sec=timeout_sec)
        return route.solved

    def render_mermaid(self, route: RetrosynthesisRoute) -> str:
        """Render route to Mermaid diagram."""
        return RouteVisualizer.to_mermaid(route)

    def render_multi_mermaid(self, routes: list) -> str:
        """Render all candidate routes to Mermaid diagrams."""
        return RouteVisualizer.to_multi_route_mermaid(routes)

    def render_comparison_table(self, routes: list) -> str:
        """Render comparative Markdown table across routes."""
        return RouteVisualizer.to_comparison_table(routes)

    def render_tree(self, route: RetrosynthesisRoute) -> str:
        """Render route to text ASCII tree."""
        return RouteVisualizer.to_text_tree(route)
