"""Multi-Step AND-OR Tree Search (Retro*) with Yield-Aware Pareto Cost Optimization."""

import heapq
import logging
import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Set

from rdkit import Chem

from tdc_studio.models.retrosynthesis.base import BaseRetroModel
from tdc_studio.models.retrosynthesis.forward_verifier import ForwardVerifier
from tdc_studio.models.retrosynthesis.rule_policy import RuleRetroPolicy
from tdc_studio.models.retrosynthesis.yield_predictor import YieldPredictor
from tdc_studio.retrosynthesis.route import ReactionStep, RetrosynthesisRoute
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

    def search(self, target_smiles: str) -> RetrosynthesisRoute:
        """Find the optimal synthetic route from target to commercial stock reagents."""
        start_time = time.time()
        canon_target = self._canonicalize(target_smiles)
        if not canon_target:
            return RetrosynthesisRoute(target_smiles=target_smiles, solved=False)

        # 0. Check if target is already in stock
        if self.stock.is_in_stock(canon_target):
            route = RetrosynthesisRoute(
                target_smiles=canon_target,
                steps=[],
                solved=True,
                starting_materials=[canon_target],
            )
            route.calculate_metrics()
            return route

        # Priority Queue for AND-OR frontier
        # Priority = cumulative step cost - log(yield)
        pq: List[SearchQueueItem] = []
        initial_item = SearchQueueItem(
            priority=0.0,
            depth=0,
            smiles=canon_target,
            path=[],
            unsolved_leaves=[canon_target],
        )
        heapq.heappush(pq, initial_item)

        visited_inchikeys: Set[str] = set()
        best_solved_route: Optional[RetrosynthesisRoute] = None
        best_cost = float("inf")

        while pq:
            if time.time() - start_time > self.timeout_sec:
                logger.info(f"Retro* search timed out after {self.timeout_sec:.1f}s")
                break

            curr = heapq.heappop(pq)

            if not curr.unsolved_leaves:
                # All leaves are solved (in stock)!
                route = RetrosynthesisRoute(
                    target_smiles=canon_target,
                    steps=curr.path,
                    solved=True,
                )
                route.calculate_metrics()
                if route.total_cost < best_cost:
                    best_cost = route.total_cost
                    best_solved_route = route
                    # Early exit on first high-quality solved route
                    break

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

            # Prevent cyclic disconnections
            key = self.stock.smiles_to_inchikey(current_target)
            if key and key in visited_inchikeys:
                continue
            if key:
                visited_inchikeys.add(key)

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

        if best_solved_route:
            return best_solved_route

        # Fallback partial route if unsolved
        fallback = RetrosynthesisRoute(
            target_smiles=canon_target,
            steps=[],
            solved=False,
        )
        fallback.calculate_metrics()
        return fallback
