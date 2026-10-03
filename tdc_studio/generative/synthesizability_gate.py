"""Hierarchical 3-Tier Synthesizability Gate for generative molecular candidates."""

import logging
from dataclasses import dataclass
from typing import Optional

from tdc_studio.generative.sa_score import calculate_sa_score
from tdc_studio.models.retrosynthesis.rule_policy import RuleRetroPolicy
from tdc_studio.retrosynthesis.planner import RetroPlanner
from tdc_studio.retrosynthesis.route import RetrosynthesisRoute
from tdc_studio.retrosynthesis.stock import StockLibrary

logger = logging.getLogger("tdc_studio.generative.synthesizability")


@dataclass
class SynthesizabilityReport:
    """Detailed synthesizability assessment report for a candidate molecule."""

    smiles: str
    passed: bool
    sa_score: float
    tier1_sa_passed: bool
    tier2_1step_passed: bool
    tier3_route_solved: bool
    route: Optional[RetrosynthesisRoute] = None
    routes: Optional[list] = None
    alternative_routes_count: int = 0
    synthetic_tractability_score: float = 0.0
    rejection_reason: Optional[str] = None


class SynthesizabilityGate:
    """3-Tier Hierarchical Synthesizability Filter for generative drug design.

    Tier 1 (Screening, 0.1ms): RDKit SAScore <= 3.5 (Heuristic molecular complexity).
    Tier 2 (Fast Feasibility, 10ms): Single-step decomposition to commercial stock precursors.
    Tier 3 (Deep Verification, 1~2s): Multi-step Retro* route search finding complete pathway.
    """

    def __init__(
        self,
        sa_threshold: float = 3.5,
        planner: Optional[RetroPlanner] = None,
        stock: Optional[StockLibrary] = None,
        rule_policy: Optional[RuleRetroPolicy] = None,
    ):
        self.sa_threshold = sa_threshold
        self.planner = planner or RetroPlanner(policy_type="rule", max_depth=4, timeout_sec=2.0)
        self.stock = stock or StockLibrary()
        self.rule_policy = rule_policy or RuleRetroPolicy()

    def evaluate_candidate(
        self, smiles: str, require_deep_route: bool = False, top_k_routes: int = 3
    ) -> SynthesizabilityReport:
        """Evaluate candidate synthesizability across the 3 tiers."""
        # --- Tier 1: SAScore Filter (0.1ms) ---
        sa_score = calculate_sa_score(smiles)
        if sa_score > self.sa_threshold:
            return SynthesizabilityReport(
                smiles=smiles,
                passed=False,
                sa_score=sa_score,
                tier1_sa_passed=False,
                tier2_1step_passed=False,
                tier3_route_solved=False,
                synthetic_tractability_score=0.0,
                rejection_reason=f"SAScore ({sa_score:.2f}) exceeds threshold ({self.sa_threshold:.2f})",
            )

        # --- Tier 2: 1-Step Stock Availability (10ms) ---
        if self.stock.is_in_stock(smiles):
            stock_route = RetrosynthesisRoute(
                target_smiles=smiles,
                steps=[],
                solved=True,
                total_depth=0,
                cumulative_yield=100.0,
                total_cost=5.0,
                starting_materials=[smiles],
            )
            return SynthesizabilityReport(
                smiles=smiles,
                passed=True,
                sa_score=sa_score,
                tier1_sa_passed=True,
                tier2_1step_passed=True,
                tier3_route_solved=True,
                route=stock_route,
                routes=[stock_route],
                alternative_routes_count=1,
                synthetic_tractability_score=1.0,
            )

        tier2_candidates = self.rule_policy.predict_reactants(smiles, top_k=5)
        tier2_passed = False
        tier2_reactants = []
        for react_str, _ in tier2_candidates:
            frags = [f.strip() for f in react_str.split(".") if f.strip()]
            all_in_stock, _ = self.stock.check_all_in_stock(frags)
            if all_in_stock:
                tier2_passed = True
                tier2_reactants = frags
                break

        # Fast feasibility tractability baseline
        tractability = 0.5 if tier2_passed else 0.2

        # If deep multi-step route is not strictly required, pass on Tier 1 + Tier 2
        if not require_deep_route:
            route_tier2 = None
            if tier2_passed:
                from tdc_studio.retrosynthesis.route import ReactionStep

                route_tier2 = RetrosynthesisRoute(
                    target_smiles=smiles,
                    steps=[
                        ReactionStep(
                            step_number=1,
                            reactants=tier2_reactants,
                            product=smiles,
                            rule_name="Single-Step Stock Disconnection",
                            confidence=0.92,
                            yield_pct=85.0,
                            cost=15.0,
                        )
                    ],
                    solved=True,
                    total_depth=1,
                    cumulative_yield=85.0,
                    total_cost=15.0,
                    starting_materials=tier2_reactants,
                )
            return SynthesizabilityReport(
                smiles=smiles,
                passed=tier2_passed,
                sa_score=sa_score,
                tier1_sa_passed=True,
                tier2_1step_passed=tier2_passed,
                tier3_route_solved=False,
                route=route_tier2,
                routes=[route_tier2] if route_tier2 else None,
                alternative_routes_count=1 if route_tier2 else 0,
                synthetic_tractability_score=round(tractability, 2),
                rejection_reason=None
                if tier2_passed
                else "No single-step stock precursor set found",
            )

        # --- Tier 3: Deep Multi-Step Retro* Route Search (1~2s) ---
        routes = self.planner.plan_routes(smiles, top_k=top_k_routes, max_depth=4, timeout_sec=2.0)
        solved_routes = [r for r in routes if r.solved]
        solved = len(solved_routes) > 0
        alt_count = len(solved_routes)

        if solved:
            # Score bonus for having multiple robust alternative pathways
            deep_tractability = 0.5 + min(0.5, alt_count * 0.15 + (0.1 if tier2_passed else 0.0))
        else:
            deep_tractability = 0.1 if tier2_passed else 0.0

        champion = solved_routes[0] if solved_routes else None

        return SynthesizabilityReport(
            smiles=smiles,
            passed=solved,
            sa_score=sa_score,
            tier1_sa_passed=True,
            tier2_1step_passed=tier2_passed,
            tier3_route_solved=solved,
            route=champion,
            routes=solved_routes if solved else None,
            alternative_routes_count=alt_count,
            synthetic_tractability_score=round(deep_tractability, 2),
            rejection_reason=None
            if solved
            else "Multi-step Retro* search could not close route to stock",
        )
