"""Unit tests for Phase 3: Stock Library, Route Data Structures, Visualizer, and Planner."""


from tdc_studio.retrosynthesis.planner import RetroPlanner
from tdc_studio.retrosynthesis.route import ReactionStep, RetrosynthesisRoute
from tdc_studio.retrosynthesis.stock import StockLibrary
from tdc_studio.retrosynthesis.visualizer import RouteVisualizer


def test_stock_library():
    stock = StockLibrary(load_builtin=True)
    assert len(stock) > 20

    # Common building blocks should be in stock
    assert stock.is_in_stock("c1ccc(C(=O)O)cc1") is True  # Benzoic acid
    assert stock.is_in_stock("CN") is True  # Methylamine
    assert stock.is_in_stock("OB(O)c1ccccc1") is True  # Phenylboronic acid

    # Novel complex target should not be in stock
    complex_target = "CC(=O)Nc1ccc(NC(=O)c2ccc(Cl)cc2)cc1"
    assert stock.is_in_stock(complex_target) is False

    all_in, status = stock.check_all_in_stock(["c1ccc(C(=O)O)cc1", "CN"])
    assert all_in is True
    assert status["CN"] is True


def test_route_data_structures():
    step1 = ReactionStep(
        step_number=1,
        reactants=["c1ccc(C(=O)O)cc1", "CN"],
        product="c1ccc(C(=O)NC)cc1",
        rule_name="Amide Coupling",
        confidence=0.95,
        yield_pct=85.0,
        cost=12.5,
    )
    route = RetrosynthesisRoute(
        target_smiles="c1ccc(C(=O)NC)cc1",
        steps=[step1],
        solved=True,
    )
    route.calculate_metrics()

    assert route.solved is True
    assert route.total_depth == 1
    assert route.cumulative_yield == 85.0
    assert route.total_cost == 12.5
    assert len(route.starting_materials) == 2

    # Serialization
    data = route.to_dict()
    assert data["target_smiles"] == "c1ccc(C(=O)NC)cc1"
    assert len(data["steps"]) == 1

    restored = RetrosynthesisRoute.from_dict(data)
    assert restored.target_smiles == route.target_smiles
    assert restored.cumulative_yield == route.cumulative_yield


def test_route_visualizer():
    step = ReactionStep(
        step_number=1,
        reactants=["c1ccc(C(=O)O)cc1", "CN"],
        product="c1ccc(C(=O)NC)cc1",
        rule_name="Amide Coupling",
        confidence=0.95,
        yield_pct=90.0,
        cost=10.0,
    )
    route = RetrosynthesisRoute(
        target_smiles="c1ccc(C(=O)NC)cc1",
        steps=[step],
        solved=True,
    )
    route.calculate_metrics()

    mermaid = RouteVisualizer.to_mermaid(route)
    assert "```mermaid" in mermaid
    assert "flowchart TD" in mermaid
    assert "Target" in mermaid
    assert "Amide Coupling" in mermaid

    text_tree = RouteVisualizer.to_text_tree(route)
    assert "Target Molecule" in text_tree
    assert "Amide Coupling" in text_tree


def test_retro_planner_solve_target():
    planner = RetroPlanner(policy_type="rule", max_depth=3, timeout_sec=3.0)

    # N-methylbenzamide is synthesized from Benzoic acid + Methylamine (both in stock!)
    target = "c1ccc(C(=O)NC)cc1"
    route = planner.plan_route(target)

    assert route.solved is True
    assert route.total_depth >= 1
    assert len(route.steps) >= 1
    assert route.cumulative_yield > 0.0

    # Test fast boolean check
    is_found = planner.is_route_found(target)
    assert is_found is True

    # Mermaid rendering helper
    mermaid = planner.render_mermaid(route)
    assert "```mermaid" in mermaid
