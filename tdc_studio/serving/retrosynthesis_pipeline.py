"""Serving pipeline for Retrosynthesis prediction and route planning."""

import logging
from typing import Optional

from tdc_studio.models.retrosynthesis.rule_policy import RuleRetroPolicy
from tdc_studio.retrosynthesis.planner import RetroPlanner
from tdc_studio.retrosynthesis.route import RetrosynthesisRoute
from tdc_studio.retrosynthesis.visualizer import RouteVisualizer
from tdc_studio.serving.schema import (
    ReactionStepSchema,
    RetroCandidateItem,
    RetroPlanResponse,
    RetroRouteItem,
    RetroSingleStepResponse,
    RouteComparisonItem,
)

logger = logging.getLogger("tdc_studio.serving.retrosynthesis")


class RetrosynthesisInferencePipeline:
    """Production serving pipeline for single-step retrosynthesis and multi-step planning."""

    def __init__(
        self,
        planner: Optional[RetroPlanner] = None,
        policy: Optional[RuleRetroPolicy] = None,
    ):
        self.planner = planner or RetroPlanner(policy_type="rule", max_depth=5, timeout_sec=5.0)
        self.single_step_policy = policy or RuleRetroPolicy()

    def predict_single_step(
        self,
        smiles: str,
        top_k: int = 5,
        reaction_type: Optional[int] = None,
    ) -> RetroSingleStepResponse:
        """Generate top-k precursor candidate sets for a target molecule."""
        candidates = self.single_step_policy.predict_reactants(
            smiles, top_k=top_k, reaction_type=reaction_type
        )
        items = [
            RetroCandidateItem(reactants=react, confidence=round(score, 4))
            for react, score in candidates
        ]
        return RetroSingleStepResponse(
            product_smiles=smiles,
            candidates=items,
            count=len(items),
        )

    def plan_route(
        self,
        smiles: str,
        top_k: int = 3,
        min_diversity: float = 0.25,
        banned_smiles: Optional[list] = None,
        max_depth: int = 5,
        timeout_sec: float = 5.0,
        render_mermaid: bool = False,
    ) -> RetroPlanResponse:
        """Find multi-step pathways to commercial stock reagents (optimal + alternative routes)."""
        routes: list = self.planner.plan_routes(
            smiles,
            top_k=top_k,
            diversity_threshold=min_diversity,
            banned_smiles=banned_smiles,
            max_depth=max_depth,
            timeout_sec=timeout_sec,
        )

        champion: RetrosynthesisRoute = routes[0] if routes else RetrosynthesisRoute(target_smiles=smiles, solved=False)

        route_items = []
        for r in routes:
            steps_schema = [
                ReactionStepSchema(
                    step_number=s.step_number,
                    reactants=s.reactants,
                    product=s.product,
                    rule_name=s.rule_name,
                    confidence=s.confidence,
                    yield_pct=s.yield_pct,
                    cost=s.cost,
                    conditions=s.conditions,
                    cost_breakdown=s.cost_breakdown,
                )
                for s in r.steps
            ]
            m_str = self.planner.render_mermaid(r) if render_mermaid else None
            route_items.append(
                RetroRouteItem(
                    rank=r.rank,
                    rank_score=r.rank_score,
                    target_smiles=r.target_smiles,
                    solved=r.solved,
                    total_depth=r.total_depth,
                    cumulative_yield=r.cumulative_yield,
                    total_cost=r.total_cost,
                    tcs_cost=r.tcs_cost,
                    synthetic_complexity_score=r.synthetic_complexity_score,
                    cost_breakdown=r.cost_breakdown,
                    starting_materials=r.starting_materials,
                    steps=steps_schema,
                    mermaid_diagram=m_str,
                )
            )

        # Comparison summary entries
        raw_summary = RouteVisualizer.to_comparison_summary(routes)
        summary_items = [RouteComparisonItem(**entry) for entry in raw_summary]

        champ_steps = [
            ReactionStepSchema(
                step_number=s.step_number,
                reactants=s.reactants,
                product=s.product,
                rule_name=s.rule_name,
                confidence=s.confidence,
                yield_pct=s.yield_pct,
                cost=s.cost,
                conditions=s.conditions,
                cost_breakdown=s.cost_breakdown,
            )
            for s in champion.steps
        ]
        champ_mermaid = self.planner.render_mermaid(champion) if render_mermaid else None

        return RetroPlanResponse(
            target_smiles=champion.target_smiles,
            solved=champion.solved,
            total_depth=champion.total_depth,
            cumulative_yield=champion.cumulative_yield,
            total_cost=champion.total_cost,
            tcs_cost=champion.tcs_cost,
            synthetic_complexity_score=champion.synthetic_complexity_score,
            cost_breakdown=champion.cost_breakdown,
            starting_materials=champion.starting_materials,
            steps=champ_steps,
            mermaid_diagram=champ_mermaid,
            routes=route_items,
            comparison_summary=summary_items,
        )

