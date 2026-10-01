"""Serving pipeline for Retrosynthesis prediction and route planning."""

import logging
from typing import Optional

from tdc_studio.models.retrosynthesis.rule_policy import RuleRetroPolicy
from tdc_studio.retrosynthesis.planner import RetroPlanner
from tdc_studio.retrosynthesis.route import RetrosynthesisRoute
from tdc_studio.serving.schema import (
    ReactionStepSchema,
    RetroCandidateItem,
    RetroPlanResponse,
    RetroSingleStepResponse,
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
        max_depth: int = 5,
        timeout_sec: float = 5.0,
        render_mermaid: bool = False,
    ) -> RetroPlanResponse:
        """Find multi-step pathway to commercial stock reagents."""
        route: RetrosynthesisRoute = self.planner.plan_route(
            smiles, max_depth=max_depth, timeout_sec=timeout_sec
        )

        steps_schema = [
            ReactionStepSchema(
                step_number=s.step_number,
                reactants=s.reactants,
                product=s.product,
                rule_name=s.rule_name,
                confidence=s.confidence,
                yield_pct=s.yield_pct,
                cost=s.cost,
            )
            for s in route.steps
        ]

        mermaid_str = None
        if render_mermaid:
            mermaid_str = self.planner.render_mermaid(route)

        return RetroPlanResponse(
            target_smiles=route.target_smiles,
            solved=route.solved,
            total_depth=route.total_depth,
            cumulative_yield=route.cumulative_yield,
            total_cost=route.total_cost,
            starting_materials=route.starting_materials,
            steps=steps_schema,
            mermaid_diagram=mermaid_str,
        )
