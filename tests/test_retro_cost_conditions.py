"""Unit tests for building block DB adapters, reaction conditions, and TCS cost engine."""

import os
import tempfile

from tdc_studio.retrosynthesis.adapters import (
    BuildingBlockRecord,
    CSVStockAdapter,
    InMemoryStockAdapter,
    SQLiteStockAdapter,
    UnifiedStockManager,
)
from tdc_studio.retrosynthesis.conditions import (
    ReactionCondition,
    ReactionConditionRecommender,
)
from tdc_studio.retrosynthesis.cost import CostBreakdown, TCSCalculator
from tdc_studio.retrosynthesis.planner import RetroPlanner
from tdc_studio.retrosynthesis.route import ReactionStep, RetrosynthesisRoute
from tdc_studio.retrosynthesis.visualizer import RouteVisualizer
from tdc_studio.serving.retrosynthesis_pipeline import (
    RetrosynthesisInferencePipeline,
)

# ------------------------------------------------------------------------------
# 1. Building Block Storage Adapters
# ------------------------------------------------------------------------------

def test_building_block_record():
    rec = BuildingBlockRecord(
        smiles="CC(=O)O",
        inchikey="QTBSBXVTEAMEQO-UHFFFAOYSA-N",
        supplier="Sigma",
        cost_per_gram=5.5,
        lead_time_days=1,
        hazards=["corrosive"],
    )
    d = rec.to_dict()
    assert d["smiles"] == "CC(=O)O"
    assert d["cost_per_gram"] == 5.5
    assert d["hazards"] == ["corrosive"]


def test_in_memory_stock_adapter():
    adapter = InMemoryStockAdapter(name="TestMemory")
    assert adapter.add_smiles("c1ccccc1", cost_per_gram=2.0, supplier="InHouse", lead_time_days=0)
    assert len(adapter) == 1

    rec = adapter.get_record("UHOVQNZJYSORNB-UHFFFAOYSA-N")
    assert rec is not None
    assert rec.smiles == "c1ccccc1"
    assert rec.cost_per_gram == 2.0
    assert rec.supplier == "InHouse"
    assert adapter.is_in_stock("UHOVQNZJYSORNB-UHFFFAOYSA-N")


def test_sqlite_stock_adapter():
    adapter = SQLiteStockAdapter(":memory:")
    records = [
        BuildingBlockRecord(
            smiles="c1ccc(O)cc1",
            inchikey="ISWSIDIOOBJBQZ-UHFFFAOYSA-N",
            supplier="Enamine",
            cost_per_gram=12.0,
            lead_time_days=3,
            hazards=["toxic", "irritant"],
        ),
        BuildingBlockRecord(
            smiles="CC(=O)Cl",
            inchikey="WETWJCDKMRYWUP-UHFFFAOYSA-N",
            supplier="Acros",
            cost_per_gram=4.5,
            lead_time_days=1,
            hazards=["air_sensitive", "corrosive"],
        ),
    ]
    inserted = adapter.load_records(records)
    assert inserted == 2
    assert len(adapter) == 2

    assert adapter.is_in_stock("ISWSIDIOOBJBQZ-UHFFFAOYSA-N")
    rec = adapter.get_record("WETWJCDKMRYWUP-UHFFFAOYSA-N")
    assert rec is not None
    assert rec.supplier == "Acros"
    assert "air_sensitive" in rec.hazards


def test_csv_stock_adapter():
    csv_content = (
        "smiles,cost_per_gram,supplier,lead_time_days\n"
        "c1ccccc1Br,15.0,VendorA,4\n"
        "OB(O)c1ccccc1,25.0,VendorB,5\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        temp_csv_path = f.name

    try:
        adapter = CSVStockAdapter(temp_csv_path)
        assert len(adapter) == 2
        # Verify bromobenzene
        assert adapter.is_in_stock("QARVLSVVCXYDNA-UHFFFAOYSA-N")
        rec = adapter.get_record("QARVLSVVCXYDNA-UHFFFAOYSA-N")
        assert rec is not None
        assert rec.cost_per_gram == 15.0
        assert rec.lead_time_days == 4
    finally:
        if os.path.exists(temp_csv_path):
            os.remove(temp_csv_path)


def test_unified_stock_manager_priorities_and_banning():
    manager = UnifiedStockManager()

    adapter_inhouse = InMemoryStockAdapter(name="InternalStock")
    adapter_inhouse.add_smiles("c1ccccc1", cost_per_gram=1.0, supplier="InHouse", lead_time_days=0)

    adapter_vendor = InMemoryStockAdapter(name="CommercialVendor")
    adapter_vendor.add_smiles("c1ccccc1", cost_per_gram=20.0, supplier="Vendor", lead_time_days=7)
    adapter_vendor.add_smiles("c1ccc(N)cc1", cost_per_gram=8.0, supplier="Vendor", lead_time_days=3)

    manager.register_adapter(adapter_inhouse, priority=1)
    manager.register_adapter(adapter_vendor, priority=2)

    # In-house record takes precedence for benzene
    rec_benzene = manager.get_record("c1ccccc1")
    assert rec_benzene is not None
    assert rec_benzene.supplier == "InHouse"
    assert rec_benzene.cost_per_gram == 1.0

    # Vendor serves aniline
    rec_aniline = manager.get_record("c1ccc(N)cc1")
    assert rec_aniline is not None
    assert rec_aniline.supplier == "Vendor"

    # Ban aniline
    assert manager.ban_compound("c1ccc(N)cc1")
    assert not manager.is_in_stock("c1ccc(N)cc1")
    assert manager.get_record("c1ccc(N)cc1") is None

    # Unban
    assert manager.unban_compound("c1ccc(N)cc1")
    assert manager.is_in_stock("c1ccc(N)cc1")


# ------------------------------------------------------------------------------
# 2. Reaction Condition Recommendation
# ------------------------------------------------------------------------------

def test_reaction_condition_recommender():
    recommender = ReactionConditionRecommender()

    # Suzuki coupling (Class 3)
    cond_suzuki = recommender.recommend_conditions(
        reactants=["c1ccc(Br)cc1", "OB(O)c1ccccc1"],
        product="c1ccc(-c2ccccc2)cc1",
        rule_name="Suzuki C-C Coupling",
    )
    assert "Pd" in str(cond_suzuki.catalyst)
    assert cond_suzuki.temperature_c >= 80.0
    assert "inert_n2" in cond_suzuki.atmosphere

    # Amide coupling (Class 2)
    cond_amide = recommender.recommend_conditions(
        reactants=["c1ccc(C(=O)O)cc1", "CN"],
        product="CNC(=O)c1ccccc1",
        rule_name="Amide Bond Formation",
    )
    assert cond_amide.catalyst is None
    assert any("HATU" in r or "EDC" in r or "DIPEA" in r for r in cond_amide.reagents)
    assert cond_amide.temperature_c == 25.0

    # Heteroatom alkylation (Class 1)
    cond_alkyl = recommender.recommend_conditions(
        reactants=["c1ccc(O)cc1", "CBr"],
        product="COc1ccccc1",
        rule_name="Ether alkylation",
    )
    assert any("K2CO3" in r for r in cond_alkyl.reagents)


# ------------------------------------------------------------------------------
# 3. Total Cost of Synthesis (TCS) & Synthetic Difficulty
# ------------------------------------------------------------------------------

def test_tcs_step_cost_calculation():
    calculator = TCSCalculator()
    cond = ReactionCondition(
        catalyst="Pd(dppf)Cl2 (3 mol%)",
        reagents=["K2CO3 (2.0 eq)"],
        solvents=["1,4-Dioxane", "H2O"],
        temperature_c=85.0,
        time_hr=4.0,
        atmosphere="inert_n2",
        purification_method="column_chromatography",
    )

    res = calculator.calculate_step_cost(
        reactants=["c1ccc(Br)cc1", "OB(O)c1ccccc1"],
        product="c1ccc(-c2ccccc2)cc1",
        yield_pct=85.0,
        condition=cond,
        reactant_costs={"c1ccc(Br)cc1": 15.0, "OB(O)c1ccccc1": 25.0},
        reactant_hazards={"c1ccc(Br)cc1": ["toxic"]},
    )
    assert res["materials_cost"] > 0
    assert res["operational_cost"] > 0  # High temp + inert atmosphere
    assert res["purification_cost"] == 35.0  # Column chromatography
    assert res["risk_penalty"] == 15.0  # Toxic hazard
    assert res["total_step_cost"] == round(
        res["materials_cost"] + res["operational_cost"] + res["purification_cost"] + res["risk_penalty"], 2
    )


def test_tcs_route_evaluation_and_scs():
    calculator = TCSCalculator()
    step1 = ReactionStep(
        step_number=1,
        reactants=["c1ccc(N)cc1", "CC(=O)Cl"],
        product="CC(=O)Nc1ccccc1",
        rule_name="Amide Coupling",
        confidence=0.95,
        yield_pct=90.0,
    )
    breakdown = calculator.evaluate_route([step1], target_smiles="CC(=O)Nc1ccccc1")
    assert isinstance(breakdown, CostBreakdown)
    assert breakdown.total_cost_per_gram > 0
    assert 1.0 <= breakdown.synthetic_complexity_score <= 10.0
    assert breakdown.estimated_lead_time_days >= 1
    assert len(breakdown.step_costs) == 1


# ------------------------------------------------------------------------------
# 4. Multi-Step Route & Visualizer Integration
# ------------------------------------------------------------------------------

def test_retrosynthesis_route_metrics_with_tcs():
    step1 = ReactionStep(
        step_number=1,
        reactants=["Nc1ccc(O)cc1", "CC(=O)O"],
        product="CC(=O)Nc1ccc(O)cc1",
        rule_name="Acylation",
        confidence=0.98,
        yield_pct=85.0,
    )
    route = RetrosynthesisRoute(
        target_smiles="CC(=O)Nc1ccc(O)cc1",
        steps=[step1],
        solved=True,
    )
    route.calculate_metrics()
    assert route.total_cost > 0
    assert route.tcs_cost > 0
    assert route.synthetic_complexity_score >= 1.0
    assert route.cost_breakdown is not None
    assert step1.cost_breakdown is not None

    # Visualization
    mermaid = RouteVisualizer.to_mermaid(route)
    assert "Acylation" in mermaid
    assert "```mermaid" in mermaid

    tree = RouteVisualizer.to_text_tree(route)
    assert "Target Molecule" in tree
    assert "Total Cost" in tree
    assert "SCS" in tree

    table = RouteVisualizer.to_comparison_table([route])
    assert "TCS" in table
    assert "SCS" in table


# ------------------------------------------------------------------------------
# 5. End-to-End RetroPlanner and Serving Pipeline Integration
# ------------------------------------------------------------------------------

def test_retro_planner_with_cost_and_conditions():
    planner = RetroPlanner(policy_type="rule", max_depth=3, timeout_sec=3.0)
    # Plan synthesis for 4-acetamidophenol (Paracetamol)
    target = "CC(=O)Nc1ccc(O)cc1"
    routes = planner.plan_routes(target, top_k=2)

    assert len(routes) >= 1
    champion = routes[0]
    assert champion.solved
    assert champion.tcs_cost > 0
    assert 1.0 <= champion.synthetic_complexity_score <= 10.0

    # Verify step conditions
    assert len(champion.steps) > 0
    first_step = champion.steps[0]
    assert first_step.conditions is not None
    assert "summary" in first_step.conditions
    assert first_step.cost_breakdown is not None


def test_serving_pipeline_with_cost_and_conditions():
    pipeline = RetrosynthesisInferencePipeline()
    res = pipeline.plan_route("CC(=O)Nc1ccc(O)cc1", top_k=1, render_mermaid=True)

    assert res.solved
    assert res.tcs_cost is not None and res.tcs_cost > 0
    assert res.synthetic_complexity_score is not None
    assert res.cost_breakdown is not None
    assert len(res.steps) > 0
    assert res.steps[0].conditions is not None
    assert res.steps[0].cost_breakdown is not None
    assert res.mermaid_diagram is not None
