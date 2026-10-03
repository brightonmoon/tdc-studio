"""Multi-Step AND-OR Tree Search (Retro*) with Yield-Aware Pareto Cost Optimization."""

import heapq
import logging
import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

from rdkit import Chem

from tdc_studio.models.retrosynthesis.base import BaseRetroModel
from tdc_studio.models.retrosynthesis.forward_verifier import ForwardVerifier
from tdc_studio.models.retrosynthesis.rule_policy import RuleRetroPolicy
from tdc_studio.models.retrosynthesis.yield_predictor import YieldPredictor
from tdc_studio.retrosynthesis.route import (
    ReactionStep,
    RetrosynthesisRoute,
    RouteDiversityEvaluator,
    RouteRanker,
)
from tdc_studio.retrosynthesis.stock import StockLibrary

logger = logging.getLogger("tdc_studio.retrosynthesis.search.retro_star")


@dataclass(order=True)
class SearchQueueItem:
    """Item for priority queue ordered by heuristic cost."""

    priority: float
    depth: int
    smiles: str = field(compare=False)
    path: List[ReactionStep] = field(compare=False, default_factory=list)
    unsolved_leaves: List[str] = field(compare=False, default_factory=list)


class RetroStarSearcher:
    """Yield-Aware Multi-Step Retrosynthesis Planner using AND-OR Tree Search."""

    def __init__(
        self,
        policy: Optional[BaseRetroModel] = None,
        stock: Optional[StockLibrary] = None,
        verifier: Optional[ForwardVerifier] = None,
        yield_predictor: Optional[YieldPredictor] = None,
        max_depth: int = 5,
        beam_width: int = 5,
        timeout_sec: float = 5.0,
        verify_round_trip: bool = False,
    ):
        self.policy = policy or RuleRetroPolicy()
        self.stock = stock or StockLibrary()
        self.verifier = verifier or ForwardVerifier()
        self.yield_predictor = yield_predictor or YieldPredictor({"hidden_dim": 64, "n_bits": 512})
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.timeout_sec = timeout_sec
        self.verify_round_trip = verify_round_trip

    def _canonicalize(self, smiles: str) -> Optional[str]:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol, canonical=True) if mol else None

    def search_top_k(
        self,
        target_smiles: str,
        top_k: int = 3,
        diversity_threshold: float = 0.25,
        banned_smiles: Optional[List[str]] = None,
        timeout_sec: Optional[float] = None,
    ) -> List[RetrosynthesisRoute]:
        """Find top-k distinct synthetic routes ordered by Pareto cost, yield, and depth."""
        timeout = timeout_sec if timeout_sec is not None else self.timeout_sec
        start_time = time.time()
        canon_target = self._canonicalize(target_smiles)
        if not canon_target:
            fallback = RetrosynthesisRoute(target_smiles=target_smiles, solved=False, rank=1)
            return [fallback]

        # Manage banned compounds dynamically
        prev_banned = set(self.stock.banned_inchikeys)
        if banned_smiles:
            for s in banned_smiles:
                self.stock.ban_compound(s)

        try:
            # 0. Check if target is already in stock
            if self.stock.is_in_stock(canon_target):
                route = RetrosynthesisRoute(
                    target_smiles=canon_target,
                    steps=[],
                    solved=True,
                    starting_materials=[canon_target],
                    rank=1,
                )
                route.calculate_metrics()
                return [route]

            # Priority Queue for AND-OR frontier
            pq: List[SearchQueueItem] = []
            initial_item = SearchQueueItem(
                priority=0.0,
                depth=0,
                smiles=canon_target,
                path=[],
                unsolved_leaves=[canon_target],
            )
            heapq.heappush(pq, initial_item)

            visited_signatures: Set[Tuple[str, ...]] = set()
            raw_solved_routes: List[RetrosynthesisRoute] = []
            seen_route_signatures: Set[str] = set()

            # Target pool size to allow diversity filtering
            target_candidates_count = max(top_k * 4, 10)

            while pq:
                if time.time() - start_time > timeout:
                    logger.info(f"Retro* search timed out after {timeout:.1f}s")
                    break

                curr = heapq.heappop(pq)

                if not curr.unsolved_leaves:
                    # All leaves are solved (in stock)!
                    route_sig = "->".join(
                        f"{s.product}:" + "+".join(sorted(s.reactants)) for s in curr.path
                    )
                    if route_sig not in seen_route_signatures:
                        seen_route_signatures.add(route_sig)
                        route = RetrosynthesisRoute(
                            target_smiles=canon_target,
                            steps=list(curr.path),
                            solved=True,
                        )
                        route.calculate_metrics()
                        raw_solved_routes.append(route)

                        if len(raw_solved_routes) >= target_candidates_count:
                            break
                    continue

                if curr.depth >= self.max_depth:
                    continue

                # Pick next molecule to expand
                current_target = curr.unsolved_leaves[0]
                remaining_leaves = curr.unsolved_leaves[1:]

                # Check if this leaf is in stock
                if self.stock.is_in_stock(current_target):
                    next_item = SearchQueueItem(
                        priority=curr.priority + self.stock.get_cost(current_target),
                        depth=curr.depth,
                        smiles=current_target,
                        path=curr.path,
                        unsolved_leaves=remaining_leaves,
                    )
                    heapq.heappush(pq, next_item)
                    continue

                # Signature to avoid expanding identical subproblems in same state
                sig = (current_target, tuple(sorted(curr.unsolved_leaves)), str(curr.depth))
                if sig in visited_signatures:
                    continue
                visited_signatures.add(sig)

                # Query policy for candidate precursors
                candidates = self.policy.predict_reactants(current_target, top_k=self.beam_width)

                for reactants_str, confidence in candidates:
                    frags = [self._canonicalize(f) for f in reactants_str.split(".") if f.strip()]
                    if not frags or any(f is None for f in frags):
                        continue

                    # Optional Round-Trip validation
                    if self.verify_round_trip:
                        valid_rt, _ = self.verifier.verify_reaction(reactants_str, current_target)
                        if not valid_rt:
                            continue

                    # Estimate reaction yield
                    pred_yield = self.yield_predictor.predict_yield(reactants_str, current_target)
                    step_cost = 10.0 + max(0.0, -math.log(max(0.05, pred_yield / 100.0)) * 5.0)

                    new_step = ReactionStep(
                        step_number=len(curr.path) + 1,
                        reactants=frags,
                        product=current_target,
                        rule_name="Disconnection",
                        confidence=confidence,
                        yield_pct=pred_yield,
                        cost=round(step_cost, 2),
                    )

                    # Formulate next unsolved leaves
                    new_leaves = list(remaining_leaves)
                    for f in frags:
                        if not self.stock.is_in_stock(f):
                            new_leaves.append(f)

                    new_priority = curr.priority + step_cost
                    next_item = SearchQueueItem(
                        priority=new_priority,
                        depth=curr.depth + 1,
                        smiles=current_target,
                        path=curr.path + [new_step],
                        unsolved_leaves=new_leaves,
                    )
                    heapq.heappush(pq, next_item)

            if raw_solved_routes:
                diverse = RouteDiversityEvaluator.filter_diverse_routes(
                    raw_solved_routes,
                    max_routes=top_k,
                    diversity_threshold=diversity_threshold,
                )
                ranked = RouteRanker.rank_routes(diverse)
                return ranked

            # Fallback partial route if unsolved
            fallback = RetrosynthesisRoute(
                target_smiles=canon_target,
                steps=[],
                solved=False,
                rank=1,
            )
            fallback.calculate_metrics()
            return [fallback]

        finally:
            self.stock.banned_inchikeys = prev_banned

    def search(self, target_smiles: str) -> RetrosynthesisRoute:
        """Find the single optimal synthetic route (backward-compatible convenience wrapper)."""
        routes = self.search_top_k(target_smiles, top_k=1)
        return routes[0]
