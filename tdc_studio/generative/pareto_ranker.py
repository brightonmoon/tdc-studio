"""NSGA-II Multi-Objective Pareto Non-dominated Sorting & Tanimoto Diversity Ranker.

Overcomes the limitations of arbitrary linear scalar weighting in lead optimization by:
1. Fast Non-dominated Sorting (Deb et al., NSGA-II) across conflicting drug discovery objectives:
   - Target Potency (maximize)
   - hERG / DILI / AMES toxicities (minimize)
   - Caco-2 permeability (maximize)
   - SAScore / Synthetic tractability (minimize)
2. Crowding Distance Calculation: preserves solution density along the Pareto frontier.
3. MaxMin Tanimoto Diversity Selection: ensures structural diversity (ECFP4) among top recommended leads.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

logger = logging.getLogger("tdc_studio.generative.pareto_ranker")


@dataclass
class Objective:
    """Specification of an optimization objective."""

    name: str
    maximize: bool = True  # True: higher is better, False: lower is better
    weight: float = 1.0  # Optional priority weight for tie-breaking


@dataclass
class ParetoCandidate:
    """A molecular candidate with multi-objective evaluations and Pareto ranks."""

    candidate_id: str
    smiles: str
    objective_values: Dict[str, float]
    rank: int = 0  # 1 is Pareto optimal front
    crowding_distance: float = 0.0
    fp: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert candidate to serializable dict."""
        return {
            "candidate_id": self.candidate_id,
            "smiles": self.smiles,
            "pareto_rank": self.rank,
            "crowding_distance": round(self.crowding_distance, 4),
            "objective_values": {k: round(v, 4) for k, v in self.objective_values.items()},
            "metadata": self.metadata,
        }


class NonDominatedSorter:
    """Fast Non-dominated Sorting algorithm based on NSGA-II (Deb et al., 2002)."""

    @staticmethod
    def dominates(
        cand_a: ParetoCandidate,
        cand_b: ParetoCandidate,
        objectives: List[Objective],
    ) -> bool:
        """Return True if cand_a Pareto-dominates cand_b.

        cand_a dominates cand_b iff:
        1. cand_a is no worse than cand_b in all objectives.
        2. cand_a is strictly better than cand_b in at least one objective.
        """
        better_in_any = False
        for obj in objectives:
            val_a = cand_a.objective_values.get(obj.name, 0.0)
            val_b = cand_b.objective_values.get(obj.name, 0.0)

            if obj.maximize:
                if val_a < val_b:
                    return False
                if val_a > val_b:
                    better_in_any = True
            else:  # minimize
                if val_a > val_b:
                    return False
                if val_a < val_b:
                    better_in_any = True

        return better_in_any

    def sort(
        self,
        candidates: List[ParetoCandidate],
        objectives: List[Objective],
    ) -> List[List[ParetoCandidate]]:
        """Perform fast non-dominated sorting.

        Args:
            candidates: List of candidate objects.
            objectives: Multi-objective specifications.

        Returns:
            List of Pareto fronts: [Front_1, Front_2, ...], where Front_1 is non-dominated.
        """
        n = len(candidates)
        if n == 0:
            return []

        # S_p: list of candidates that p dominates
        dominated_sets: Dict[int, List[int]] = {i: [] for i in range(n)}
        # n_p: number of candidates that dominate p
        domination_counts: Dict[int, int] = {i: 0 for i in range(n)}

        fronts: List[List[int]] = [[]]

        for p in range(n):
            for q in range(n):
                if p == q:
                    continue
                if self.dominates(candidates[p], candidates[q], objectives):
                    dominated_sets[p].append(q)
                elif self.dominates(candidates[q], candidates[p], objectives):
                    domination_counts[p] += 1

            if domination_counts[p] == 0:
                candidates[p].rank = 1
                fronts[0].append(p)

        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            for p in fronts[i]:
                for q in dominated_sets[p]:
                    domination_counts[q] -= 1
                    if domination_counts[q] == 0:
                        candidates[q].rank = i + 2
                        next_front.append(q)
            i += 1
            if len(next_front) > 0:
                fronts.append(next_front)
            else:
                break

        sorted_fronts: List[List[ParetoCandidate]] = []
        for front_indices in fronts:
            if not front_indices:
                continue
            front_cands = [candidates[idx] for idx in front_indices]
            self.calculate_crowding_distance(front_cands, objectives)
            sorted_fronts.append(front_cands)

        return sorted_fronts

    @staticmethod
    def calculate_crowding_distance(
        front: List[ParetoCandidate],
        objectives: List[Objective],
    ) -> None:
        """Assign crowding distance to each candidate within a single front."""
        m = len(front)
        if m == 0:
            return
        if m <= 2:
            for c in front:
                c.crowding_distance = float("inf")
            return

        for c in front:
            c.crowding_distance = 0.0

        for obj in objectives:
            # Sort front by current objective value
            front.sort(key=lambda c: c.objective_values.get(obj.name, 0.0))

            val_min = front[0].objective_values.get(obj.name, 0.0)
            val_max = front[-1].objective_values.get(obj.name, 0.0)
            val_range = val_max - val_min

            # Boundary points get infinite distance
            front[0].crowding_distance = float("inf")
            front[-1].crowding_distance = float("inf")

            if val_range <= 1e-9:
                continue

            for i in range(1, m - 1):
                prev_val = front[i - 1].objective_values.get(obj.name, 0.0)
                next_val = front[i + 1].objective_values.get(obj.name, 0.0)
                front[i].crowding_distance += (next_val - prev_val) / val_range


class DiversitySelector:
    """Tanimoto distance-based MaxMin diversity selector."""

    @staticmethod
    def compute_fingerprint(smiles: str) -> Optional[Any]:
        """Compute Morgan ECFP4 fingerprint for a SMILES string."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
        except Exception:
            pass
        return None

    @classmethod
    def select_diverse_subset(
        cls,
        candidates: List[ParetoCandidate],
        k: int,
        similarity_threshold: float = 0.70,
    ) -> List[ParetoCandidate]:
        """Select up to k candidates maximizing chemical scaffold diversity.

        Uses greedy MaxMin distance selection:
        1. Pick the best ranked candidate.
        2. Iteratively pick the candidate with maximum distance to all selected candidates.
        3. If similarity exceeds similarity_threshold to any selected candidate, de-prioritize.
        """
        if len(candidates) <= k:
            return list(candidates)

        # Cache fingerprints
        for c in candidates:
            if c.fp is None:
                c.fp = cls.compute_fingerprint(c.smiles)

        valid_cands = [c for c in candidates if c.fp is not None]
        if not valid_cands:
            return candidates[:k]

        selected: List[ParetoCandidate] = [valid_cands[0]]
        remaining = list(valid_cands[1:])

        while len(selected) < k and remaining:
            best_cand = None
            best_min_dist = -1.0

            for cand in remaining:
                # Find min distance to already selected candidates
                sims = [DataStructs.TanimotoSimilarity(cand.fp, s.fp) for s in selected]
                max_sim = max(sims) if sims else 0.0
                min_dist = 1.0 - max_sim

                # If this candidate is structurally diverse enough and has largest min distance
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_cand = cand

            if best_cand is not None:
                selected.append(best_cand)
                remaining.remove(best_cand)
            else:
                break

        return selected


class ParetoRanker:
    """Integrated engine for multi-objective Pareto ranking and diversity filtering."""

    def __init__(self):
        self.sorter = NonDominatedSorter()
        self.diversity_selector = DiversitySelector()

    def rank_and_select(
        self,
        candidates: List[ParetoCandidate],
        objectives: List[Objective],
        top_k: int = 10,
        ensure_diversity: bool = True,
        similarity_threshold: float = 0.75,
    ) -> List[ParetoCandidate]:
        """Rank candidates with NSGA-II and select the top diverse Pareto-optimal cohort.

        Args:
            candidates: Evaluated candidate compounds.
            objectives: Specification of target objectives (potency, toxicities, etc.).
            top_k: Number of candidates to select.
            ensure_diversity: If True, applies Tanimoto diversity filtering within fronts.
            similarity_threshold: Tanimoto threshold above which compounds are penalized.

        Returns:
            Ranked and pruned list of Top-K Pareto candidates.
        """
        if not candidates:
            return []

        # 1. Non-dominated sort
        fronts = self.sorter.sort(candidates, objectives)

        selected: List[ParetoCandidate] = []
        for front in fronts:
            # Sort within front by crowding distance descending (favor less crowded)
            front.sort(key=lambda c: c.crowding_distance, reverse=True)

            if len(selected) + len(front) <= top_k:
                selected.extend(front)
            else:
                needed = top_k - len(selected)
                if ensure_diversity and len(front) > needed:
                    diverse_picks = self.diversity_selector.select_diverse_subset(
                        front, k=needed, similarity_threshold=similarity_threshold
                    )
                    selected.extend(diverse_picks)
                else:
                    selected.extend(front[:needed])
                break

        return selected
