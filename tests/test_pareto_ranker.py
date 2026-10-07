"""Unit tests for NSGA-II ParetoRanker and DiversitySelector."""


from tdc_studio.generative.pareto_ranker import (
    DiversitySelector,
    NonDominatedSorter,
    Objective,
    ParetoCandidate,
    ParetoRanker,
)


def test_pareto_domination():
    """Test non-domination comparison logic."""
    objectives = [
        Objective(name="potency", maximize=True),
        Objective(name="herg_risk", maximize=False),
    ]

    # cand_a is strictly better in potency and no worse in herg
    cand_a = ParetoCandidate(
        candidate_id="A",
        smiles="CC(=O)O",
        objective_values={"potency": 8.5, "herg_risk": 0.2},
    )
    cand_b = ParetoCandidate(
        candidate_id="B",
        smiles="CC(=O)N",
        objective_values={"potency": 7.0, "herg_risk": 0.5},
    )

    assert NonDominatedSorter.dominates(cand_a, cand_b, objectives) is True
    assert NonDominatedSorter.dominates(cand_b, cand_a, objectives) is False

    # Trade-off: cand_c has higher potency but worse herg than cand_a -> neither dominates
    cand_c = ParetoCandidate(
        candidate_id="C",
        smiles="CC(=O)Cl",
        objective_values={"potency": 9.5, "herg_risk": 0.4},
    )
    assert NonDominatedSorter.dominates(cand_a, cand_c, objectives) is False
    assert NonDominatedSorter.dominates(cand_c, cand_a, objectives) is False


def test_fast_non_dominated_sort():
    """Test sorting candidates into Pareto fronts."""
    objectives = [
        Objective(name="potency", maximize=True),
        Objective(name="sa_score", maximize=False),
    ]

    candidates = [
        # Front 1 candidates
        ParetoCandidate("C1", "c1ccccc1", {"potency": 9.0, "sa_score": 2.0}),
        ParetoCandidate("C2", "c1ccncc1", {"potency": 8.0, "sa_score": 1.5}),
        # Dominated by C1 (lower potency, higher SA)
        ParetoCandidate("C3", "c1ccc(Cl)cc1", {"potency": 7.0, "sa_score": 3.0}),
    ]

    sorter = NonDominatedSorter()
    fronts = sorter.sort(candidates, objectives)

    assert len(fronts) >= 2
    front1_ids = [c.candidate_id for c in fronts[0]]
    assert "C1" in front1_ids
    assert "C2" in front1_ids
    assert "C3" not in front1_ids

    # Front 1 candidates should have rank 1
    for c in fronts[0]:
        assert c.rank == 1


def test_diversity_selector():
    """Test structural diversity filtering among candidates."""
    # Create two nearly identical benzene derivatives and one distinct morpholine
    candidates = [
        ParetoCandidate("1", "c1ccccc1C", {"val": 1.0}),  # Toluene
        ParetoCandidate("2", "c1ccccc1CC", {"val": 1.0}),  # Ethylbenzene (similar)
        ParetoCandidate("3", "C1COCCN1", {"val": 1.0}),  # Morpholine (different)
    ]

    # Select top 2 diverse candidates
    diverse = DiversitySelector.select_diverse_subset(candidates, k=2, similarity_threshold=0.6)
    assert len(diverse) == 2
    picked_ids = [c.candidate_id for c in diverse]
    assert "1" in picked_ids
    # The distinct morpholine should be selected over the similar ethylbenzene
    assert "3" in picked_ids


def test_pareto_ranker_integration():
    """Test end-to-end Pareto ranking and selection."""
    ranker = ParetoRanker()
    objectives = [
        Objective(name="delta", maximize=True),
        Objective(name="sa_score", maximize=False),
    ]

    candidates = [
        ParetoCandidate("A", "c1ccccc1", {"delta": 0.4, "sa_score": 2.2}),
        ParetoCandidate("B", "c1ccncc1", {"delta": 0.5, "sa_score": 2.5}),
        ParetoCandidate("C", "c1ncccn1", {"delta": 0.1, "sa_score": 4.5}),
        ParetoCandidate("D", "C1CCCCC1", {"delta": 0.3, "sa_score": 2.0}),
    ]

    top_2 = ranker.rank_and_select(candidates, objectives, top_k=2, ensure_diversity=False)
    assert len(top_2) == 2
    # Candidate C is strictly dominated and should not be in top 2
    picked_ids = [c.candidate_id for c in top_2]
    assert "C" not in picked_ids
