"""Unit tests for Retrosynthesis Top-K alternative routes, diversity filtering, and stock constraints."""

from tdc_studio.generative.synthesizability_gate import SynthesizabilityGate
from tdc_studio.retrosynthesis.planner import RetroPlanner
from tdc_studio.retrosynthesis.route import (
    ReactionStep,
    RetrosynthesisRoute,
    RouteDiversityEvaluator,
    RouteRanker,
)
from tdc_studio.retrosynthesis.stock import StockLibrary
from tdc_studio.retrosynthesis.visualizer import RouteVisualizer
from tdc_studio.serving.retrosynthesis_pipeline import RetrosynthesisInferencePipeline


def test_route_diversity_evaluator():
    # Route 1: Amide from Benzoic acid + Methylamine
    r1 = RetrosynthesisRoute(
        target_smiles="c1ccc(C(=O)NC)cc1",
        steps=[
            ReactionStep(
                1,
                ["c1ccc(C(=O)O)cc1", "CN"],
                "c1ccc(C(=O)NC)cc1",
                "Amide Coupling",
                0.95,
                85.0,
                10.0,
            )
        ],
        solved=True,
    )
    r1.calculate_metrics()

    # Route 2: Same reaction, slightly different ordering
    r2 = RetrosynthesisRoute(
        target_smiles="c1ccc(C(=O)NC)cc1",
        steps=[
            ReactionStep(
                1,
                ["CN", "c1ccc(C(=O)O)cc1"],
                "c1ccc(C(=O)NC)cc1",
                "Amide Coupling",
                0.92,
                80.0,
                11.0,
            )
        ],
        solved=True,
    )
    r2.calculate_metrics()

    # Route 3: Alternative reaction: Acid Chloride + Methylamine
    r3 = RetrosynthesisRoute(
        target_smiles="c1ccc(C(=O)NC)cc1",
        steps=[
            ReactionStep(
                1,
                ["c1ccc(C(=O)Cl)cc1", "CN"],
                "c1ccc(C(=O)NC)cc1",
                "Acid Chloride Amidation",
                0.98,
                92.0,
                15.0,
            )
        ],
        solved=True,
    )
    r3.calculate_metrics()

    # Similarity between r1 and r2 should be high (distance low)
    dist_12 = RouteDiversityEvaluator.compute_distance(r1, r2)
    assert dist_12 < 0.15

    # Distance between r1 and r3 should be significant
    dist_13 = RouteDiversityEvaluator.compute_distance(r1, r3)
    assert dist_13 >= 0.20

    # Diversity filtering should pick r1 and r3, skipping r2
    diverse = RouteDiversityEvaluator.filter_diverse_routes(
        [r1, r2, r3], max_routes=2, diversity_threshold=0.20
    )
    assert len(diverse) == 2
    assert r1 in diverse
    assert r3 in diverse


def test_route_ranker():
    r1 = RetrosynthesisRoute(
        target_smiles="c1ccccc1",
        steps=[ReactionStep(1, ["A", "B"], "c1ccccc1", "Rule1", 0.9, 85.0, 15.0)],
        solved=True,
    )
    r2 = RetrosynthesisRoute(
        target_smiles="c1ccccc1",
        steps=[ReactionStep(1, ["C", "D"], "c1ccccc1", "Rule2", 0.95, 95.0, 8.0)],
        solved=True,
    )

    ranked = RouteRanker.rank_routes([r1, r2])
    # r2 has lower cost ($8 vs $15) and higher yield (95% vs 85%), so r2 should be Rank 1!
    assert ranked[0].rank == 1
    assert ranked[0].total_cost == 8.0
    assert ranked[1].rank == 2
    assert ranked[1].total_cost == 15.0


def test_stock_ban_and_unban():
    stock = StockLibrary(load_builtin=True)
    benzoic_acid = "c1ccc(C(=O)O)cc1"
    assert stock.is_in_stock(benzoic_acid) is True

    # Ban benzoic acid
    stock.ban_compound(benzoic_acid)
    assert stock.is_banned(benzoic_acid) is True
    assert stock.is_in_stock(benzoic_acid) is False

    # Unban benzoic acid
    stock.unban_compound(benzoic_acid)
    assert stock.is_banned(benzoic_acid) is False
    assert stock.is_in_stock(benzoic_acid) is True


def test_planner_plan_routes_top_k():
    planner = RetroPlanner(policy_type="rule", max_depth=3, timeout_sec=4.0)
    target = "c1ccc(C(=O)NC)cc1"

    # Search top-k routes
    routes = planner.plan_routes(target, top_k=3, diversity_threshold=0.15)
    assert len(routes) >= 1
    assert routes[0].solved is True
    assert routes[0].rank == 1

    # Check metrics
    for r in routes:
        assert r.total_depth >= 1
        assert r.cumulative_yield > 0.0


def test_planner_banned_smiles_rerouting():
    from rdkit import Chem

    planner = RetroPlanner(policy_type="rule", max_depth=3, timeout_sec=4.0)
    target = "c1ccc(C(=O)NC)cc1"
    canon_acid = Chem.CanonSmiles("c1ccc(C(=O)O)cc1")

    # First find default route (uses benzoic acid + methylamine)
    default_route = planner.plan_route(target)
    assert default_route.solved is True
    assert canon_acid in default_route.starting_materials

    # Ban benzoic acid to simulate supply disruption
    banned_route = planner.plan_routes(
        target,
        top_k=1,
        banned_smiles=["c1ccc(C(=O)O)cc1"],
    )[0]

    # The banned compound should NOT appear in starting materials
    assert canon_acid not in banned_route.starting_materials


def test_visualizer_comparison_table_and_multi_mermaid():
    r1 = RetrosynthesisRoute(
        target_smiles="c1ccc(C(=O)NC)cc1",
        steps=[
            ReactionStep(
                1,
                ["c1ccc(C(=O)O)cc1", "CN"],
                "c1ccc(C(=O)NC)cc1",
                "Amide Coupling",
                0.95,
                85.0,
                10.0,
            )
        ],
        solved=True,
        rank=1,
    )
    r1.calculate_metrics()

    r2 = RetrosynthesisRoute(
        target_smiles="c1ccc(C(=O)NC)cc1",
        steps=[
            ReactionStep(
                1,
                ["c1ccc(C(=O)Cl)cc1", "CN"],
                "c1ccc(C(=O)NC)cc1",
                "Acid Chloride Coupling",
                0.90,
                80.0,
                14.0,
            )
        ],
        solved=True,
        rank=2,
    )
    r2.calculate_metrics()

    table = RouteVisualizer.to_comparison_table([r1, r2])
    assert "순위 (Rank)" in table
    assert "🥇 1위 (최적)" in table
    assert "🥈 2위 (대안 A)" in table

    multi_mermaid = RouteVisualizer.to_multi_route_mermaid([r1, r2])
    assert "1순위 최적 경로" in multi_mermaid
    assert "2순위 대안 경로" in multi_mermaid
    assert "```mermaid" in multi_mermaid


def test_serving_pipeline_multi_route():
    pipe = RetrosynthesisInferencePipeline()
    resp = pipe.plan_route("c1ccc(C(=O)NC)cc1", top_k=2, render_mermaid=True)

    assert resp.solved is True
    assert resp.total_depth >= 1
    assert len(resp.routes) >= 1
    assert resp.routes[0].rank == 1
    assert len(resp.comparison_summary) >= 1
    assert resp.mermaid_diagram is not None


def test_synthesizability_gate_with_tractability():
    gate = SynthesizabilityGate()
    target = "c1ccc(C(=O)NC)cc1"

    report = gate.evaluate_candidate(target, require_deep_route=True, top_k_routes=2)
    assert report.passed is True
    assert report.tier3_route_solved is True
    assert report.alternative_routes_count >= 1
    assert 0.0 < report.synthetic_tractability_score <= 1.0
