"""Total Cost of Synthesis (TCS) and Synthetic Difficulty Engine.

Models realistic raw material costs, cumulative yield losses, operational harshness,
purification complexity, and safety risks.
"""

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from rdkit import Chem
from rdkit.Chem import Descriptors

from tdc_studio.retrosynthesis.conditions import ReactionCondition

logger = logging.getLogger("tdc_studio.retrosynthesis.cost")


@dataclass
class CostBreakdown:
    """Detailed financial and operational cost breakdown for a synthetic pathway."""

    raw_materials_cost: float = 0.0      # Material cost per gram of target ($/g)
    operational_cost: float = 0.0          # Equipment, temperature, inert atmosphere cost ($/g)
    purification_cost: float = 0.0         # Chromatography/isolation cost ($/g)
    safety_risk_penalty: float = 0.0       # Toxic/pyrophoric handling surcharge ($/g)
    total_cost_per_gram: float = 0.0       # Total estimated synthesis cost ($/g)
    synthetic_complexity_score: float = 1.0# Normalized difficulty index (1.0 = easy, 10.0 = very hard)
    estimated_lead_time_days: int = 2     # Total procurement + synthesis lead time (days)
    step_costs: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TCSCalculator:
    """Total Cost of Synthesis (TCS) and Operational Difficulty Calculator."""

    def __init__(self, currency: str = "USD"):
        self.currency = currency

    @staticmethod
    def get_molecular_weight(smiles: str) -> float:
        """Calculate molecular weight using RDKit."""
        mol = Chem.MolFromSmiles(smiles)
        return Descriptors.MolWt(mol) if mol else 200.0

    def calculate_step_cost(
        self,
        reactants: List[str],
        product: str,
        yield_pct: float,
        condition: Optional[Any] = None,
        reactant_costs: Optional[Dict[str, float]] = None,
        reactant_hazards: Optional[Dict[str, List[str]]] = None,
        cumulative_yield_downstream: float = 1.0,
    ) -> Dict[str, Any]:
        """Compute the itemized synthesis cost for a single reaction step."""
        if isinstance(condition, dict):
            cond = ReactionCondition(**condition)
        else:
            cond = condition or ReactionCondition()
        r_costs = reactant_costs or {}
        r_hazards = reactant_hazards or {}

        mw_prod = self.get_molecular_weight(product)
        yield_ratio = max(0.05, yield_pct / 100.0)
        yield_loss_multiplier = 1.0 / (yield_ratio * max(0.01, cumulative_yield_downstream))

        # 1. Raw Material Costs
        mat_cost = 0.0
        for r in reactants:
            mw_r = self.get_molecular_weight(r)
            stoich_factor = (mw_r / max(1.0, mw_prod)) * yield_loss_multiplier
            unit_price = r_costs.get(r, 10.0)
            mat_cost += unit_price * stoich_factor

        # Catalyst & Reagent consumables cost
        if cond.catalyst:
            mat_cost += 18.0  # Precious metal catalyst (Pd, Rh, Ir) amortized cost
        mat_cost += len(cond.reagents) * 3.5
        mat_cost += len(cond.solvents) * 2.0

        # 2. Operational & Equipment Costs
        op_cost = 0.0
        # Temperature penalty
        if cond.temperature_c <= -40.0:
            op_cost += 30.0  # Cryogenic liquid N2/dry ice
        elif cond.temperature_c >= 80.0:
            op_cost += 10.0  # Heating mantle / reflux condenser
        elif cond.temperature_c == 0.0:
            op_cost += 4.0   # Ice bath

        # Atmosphere penalty
        if cond.atmosphere == "inert_n2":
            op_cost += 12.0  # Schlenk line & N2 gas purge
        elif cond.atmosphere == "glovebox":
            op_cost += 45.0  # Inert atmosphere glovebox chamber

        # Duration labor
        if cond.time_hr >= 8.0:
            op_cost += 8.0   # Overnight monitoring

        # 3. Purification Cost
        purif_cost = 0.0
        p_method = cond.purification_method.lower()
        if "filtration" in p_method:
            purif_cost += 4.0
        elif "recrystallization" in p_method:
            purif_cost += 8.0
        elif "column" in p_method:
            purif_cost += 35.0  # Silica gel column + 1L solvent consumption
        elif "hplc" in p_method:
            purif_cost += 85.0  # Prep-HPLC solvent & column wear

        # 4. Safety Risk Surcharges
        risk_penalty = 0.0
        for r in reactants:
            hazards = r_hazards.get(r, [])
            for h in hazards:
                if "pyrophoric" in h or "explosive" in h:
                    risk_penalty += 35.0
                elif "toxic" in h or "air_sensitive" in h:
                    risk_penalty += 15.0

        step_total = mat_cost + op_cost + purif_cost + risk_penalty

        return {
            "materials_cost": round(mat_cost, 2),
            "operational_cost": round(op_cost, 2),
            "purification_cost": round(purif_cost, 2),
            "risk_penalty": round(risk_penalty, 2),
            "total_step_cost": round(step_total, 2),
            "condition": cond.summary,
            "purification_method": cond.purification_method,
        }

    def evaluate_route(
        self,
        steps: List[Any],  # List[ReactionStep]
        target_smiles: str,
        stock_manager: Optional[Any] = None,
    ) -> CostBreakdown:
        """Compute the full route Total Cost of Synthesis (TCS) and complexity index."""
        if not steps:
            return CostBreakdown(total_cost_per_gram=10.0, synthetic_complexity_score=1.0)

        tot_materials = 0.0
        tot_operational = 0.0
        tot_purification = 0.0
        tot_risk = 0.0
        step_records = []
        max_lead_time = 0
        harsh_conditions = 0

        # Compute cumulative yield product downstream from each step
        n_steps = len(steps)
        downstream_yields = [1.0] * n_steps
        running_yield = 1.0
        for i in reversed(range(n_steps - 1)):
            running_yield *= (steps[i + 1].yield_pct / 100.0)
            downstream_yields[i] = running_yield

        for idx, s in enumerate(steps):
            cond = getattr(s, "conditions", None)
            if not cond:
                from tdc_studio.retrosynthesis.conditions import ReactionConditionRecommender

                recommender = ReactionConditionRecommender()
                cond = recommender.recommend_conditions(s.reactants, s.product, rule_name=s.rule_name)
            elif isinstance(cond, dict):
                cond = ReactionCondition(**cond)

            # Retrieve metadata from stock manager if available
            r_costs = {}
            r_hazards = {}
            if stock_manager:
                for r in s.reactants:
                    rec = stock_manager.get_record(r)
                    if rec:
                        r_costs[r] = rec.cost_per_gram
                        r_hazards[r] = rec.hazards
                        max_lead_time = max(max_lead_time, rec.lead_time_days)
                    else:
                        r_costs[r] = 20.0  # Estimated intermediate cost

            step_res = self.calculate_step_cost(
                reactants=s.reactants,
                product=s.product,
                yield_pct=s.yield_pct,
                condition=cond,
                reactant_costs=r_costs,
                reactant_hazards=r_hazards,
                cumulative_yield_downstream=downstream_yields[idx],
            )

            tot_materials += step_res["materials_cost"]
            tot_operational += step_res["operational_cost"]
            tot_purification += step_res["purification_cost"]
            tot_risk += step_res["risk_penalty"]
            step_records.append(step_res)

            if cond.temperature_c <= -40.0 or cond.temperature_c >= 80.0 or cond.atmosphere != "air":
                harsh_conditions += 1

        grand_total = tot_materials + tot_operational + tot_purification + tot_risk

        # Synthetic Complexity Score (SCS: 1.0 ~ 10.0 scale)
        depth = len(steps)
        cum_yield = max(0.01, downstream_yields[0] * (steps[-1].yield_pct / 100.0))
        yield_penalty = (1.0 - cum_yield) * 3.0
        harsh_penalty = harsh_conditions * 0.8
        raw_scs = 1.0 + (depth * 0.8) + yield_penalty + harsh_penalty
        scs = round(min(10.0, max(1.0, raw_scs)), 2)

        # Estimated total lead time: material lead time + 1 day per synthesis/purification step
        est_lead_time = max_lead_time + (depth * 1)

        return CostBreakdown(
            raw_materials_cost=round(tot_materials, 2),
            operational_cost=round(tot_operational, 2),
            purification_cost=round(tot_purification, 2),
            safety_risk_penalty=round(tot_risk, 2),
            total_cost_per_gram=round(grand_total, 2),
            synthetic_complexity_score=scs,
            estimated_lead_time_days=est_lead_time,
            step_costs=step_records,
        )
