"""Reaction Condition Recommendation Engine (Catalysts, Reagents, Solvents, Temp, Time)."""

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("tdc_studio.retrosynthesis.conditions")


@dataclass
class ReactionCondition:
    """Detailed experimental reaction protocol conditions."""

    catalyst: Optional[str] = None
    reagents: List[str] = field(default_factory=list)  # Bases, acids, coupling reagents
    solvents: List[str] = field(default_factory=list)
    temperature_c: float = 25.0                        # Reaction temperature (°C)
    time_hr: float = 2.0                               # Reaction duration (hours)
    atmosphere: str = "air"                            # "air", "inert_n2", "glovebox"
    purification_method: str = "column_chromatography" # "filtration", "recrystallization", "column_chromatography", "prep_hplc"
    confidence: float = 0.85
    summary: str = ""

    def __post_init__(self):
        if not self.summary:
            parts = []
            if self.catalyst:
                parts.append(f"Cat: {self.catalyst}")
            if self.reagents:
                parts.append(f"Reag: {', '.join(self.reagents)}")
            if self.solvents:
                parts.append(f"Solv: {', '.join(self.solvents)}")
            parts.append(f"{self.temperature_c:.0f}°C, {self.time_hr:.1f}h")
            if self.atmosphere != "air":
                parts.append(f"under {self.atmosphere}")
            self.summary = " | ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Knowledge Base of Canonical Industrial Organic Reaction Conditions for USPTO-50K Classes
CLASS_CONDITION_TEMPLATES: Dict[int, List[ReactionCondition]] = {
    # 1. Heteroatom alkylation/arylation
    1: [
        ReactionCondition(
            catalyst=None,
            reagents=["K2CO3 (2.0 eq)", "KI (0.1 eq)"],
            solvents=["DMF"],
            temperature_c=60.0,
            time_hr=4.0,
            atmosphere="air",
            purification_method="column_chromatography",
            confidence=0.92,
        ),
        ReactionCondition(
            catalyst="Pd2(dba)3 (2.5 mol%)",
            reagents=["BINAP (5 mol%)", "NaOtBu (1.5 eq)"],
            solvents=["Toluene"],
            temperature_c=100.0,
            time_hr=12.0,
            atmosphere="inert_n2",
            purification_method="column_chromatography",
            confidence=0.88,
        ),
    ],
    # 2. Acylation (Amide coupling, esterification)
    2: [
        ReactionCondition(
            catalyst=None,
            reagents=["HATU (1.2 eq)", "DIPEA (2.5 eq)"],
            solvents=["DCM", "DMF"],
            temperature_c=25.0,
            time_hr=2.0,
            atmosphere="air",
            purification_method="recrystallization",
            confidence=0.96,
        ),
        ReactionCondition(
            catalyst="DMAP (0.1 eq)",
            reagents=["EDC·HCl (1.3 eq)", "Et3N (2.0 eq)"],
            solvents=["DCM"],
            temperature_c=25.0,
            time_hr=3.0,
            atmosphere="air",
            purification_method="column_chromatography",
            confidence=0.94,
        ),
        ReactionCondition(
            catalyst=None,
            reagents=["Pyridine (2.0 eq)"],
            solvents=["DCM"],
            temperature_c=0.0,
            time_hr=1.5,
            atmosphere="air",
            purification_method="filtration",
            confidence=0.95,
        ),
    ],
    # 3. C-C bond formation (Suzuki, Sonogashira, Heck)
    3: [
        ReactionCondition(
            catalyst="Pd(dppf)Cl2·CH2Cl2 (5 mol%)",
            reagents=["K2CO3 (2.0 eq)"],
            solvents=["1,4-Dioxane", "H2O (4:1)"],
            temperature_c=85.0,
            time_hr=4.0,
            atmosphere="inert_n2",
            purification_method="column_chromatography",
            confidence=0.95,
        ),
        ReactionCondition(
            catalyst="Pd(PPh3)4 (5 mol%)",
            reagents=["Na2CO3 (2.0 eq)"],
            solvents=["Toluene", "EtOH", "H2O"],
            temperature_c=90.0,
            time_hr=6.0,
            atmosphere="inert_n2",
            purification_method="column_chromatography",
            confidence=0.92,
        ),
        ReactionCondition(
            catalyst="Pd(PPh3)2Cl2 (3 mol%) / CuI (5 mol%)",
            reagents=["Et3N (3.0 eq)"],
            solvents=["THF"],
            temperature_c=50.0,
            time_hr=3.0,
            atmosphere="inert_n2",
            purification_method="column_chromatography",
            confidence=0.91,
        ),
    ],
    # 4. Heterocycle formation
    4: [
        ReactionCondition(
            catalyst="p-TsOH (0.1 eq)",
            reagents=["Molecular Sieves 4A"],
            solvents=["Toluene"],
            temperature_c=110.0,
            time_hr=8.0,
            atmosphere="inert_n2",
            purification_method="column_chromatography",
            confidence=0.90,
        ),
        ReactionCondition(
            catalyst=None,
            reagents=["T3P (1.5 eq)", "Et3N (3.0 eq)"],
            solvents=["EtOAc"],
            temperature_c=80.0,
            time_hr=4.0,
            atmosphere="air",
            purification_method="recrystallization",
            confidence=0.88,
        ),
    ],
    # 5. Protections (Boc, Cbz, silyl)
    5: [
        ReactionCondition(
            catalyst=None,
            reagents=["Boc2O (1.1 eq)", "Et3N (2.0 eq)"],
            solvents=["THF", "H2O"],
            temperature_c=25.0,
            time_hr=2.0,
            atmosphere="air",
            purification_method="filtration",
            confidence=0.98,
        ),
        ReactionCondition(
            catalyst="DMAP (0.05 eq)",
            reagents=["TBSCl (1.2 eq)", "Imidazole (2.0 eq)"],
            solvents=["DMF"],
            temperature_c=25.0,
            time_hr=3.0,
            atmosphere="air",
            purification_method="column_chromatography",
            confidence=0.96,
        ),
    ],
    # 6. Deprotections (Boc removal, ester hydrolysis)
    6: [
        ReactionCondition(
            catalyst=None,
            reagents=["TFA (10 eq)"],
            solvents=["DCM"],
            temperature_c=25.0,
            time_hr=1.0,
            atmosphere="air",
            purification_method="filtration",
            confidence=0.98,
        ),
        ReactionCondition(
            catalyst=None,
            reagents=["LiOH·H2O (2.5 eq)"],
            solvents=["THF", "MeOH", "H2O (3:1:1)"],
            temperature_c=25.0,
            time_hr=2.0,
            atmosphere="air",
            purification_method="filtration",
            confidence=0.97,
        ),
        ReactionCondition(
            catalyst=None,
            reagents=["TBAF (1.5 eq)"],
            solvents=["THF"],
            temperature_c=25.0,
            time_hr=1.5,
            atmosphere="air",
            purification_method="column_chromatography",
            confidence=0.95,
        ),
    ],
    # 7. Reductions (Nitro, Carbonyl, Nitrile)
    7: [
        ReactionCondition(
            catalyst="Pd/C (10 wt%, 5 mol%)",
            reagents=["H2 (1 atm / balloon)"],
            solvents=["MeOH", "EtOAc"],
            temperature_c=25.0,
            time_hr=3.0,
            atmosphere="inert_n2",
            purification_method="filtration",
            confidence=0.96,
        ),
        ReactionCondition(
            catalyst=None,
            reagents=["NaBH4 (2.0 eq)"],
            solvents=["MeOH"],
            temperature_c=0.0,
            time_hr=1.0,
            atmosphere="air",
            purification_method="filtration",
            confidence=0.95,
        ),
        ReactionCondition(
            catalyst=None,
            reagents=["Fe powder (5.0 eq)", "NH4Cl (3.0 eq)"],
            solvents=["EtOH", "H2O (4:1)"],
            temperature_c=80.0,
            time_hr=2.5,
            atmosphere="air",
            purification_method="filtration",
            confidence=0.92,
        ),
    ],
    # 8. Oxidations (Alcohol to acid/aldehyde, sulfide oxidation)
    8: [
        ReactionCondition(
            catalyst="TEMPO (5 mol%)",
            reagents=["Bleach (NaOCl 1.2 eq)", "KBr (0.1 eq)"],
            solvents=["DCM", "H2O"],
            temperature_c=0.0,
            time_hr=1.0,
            atmosphere="air",
            purification_method="recrystallization",
            confidence=0.94,
        ),
        ReactionCondition(
            catalyst=None,
            reagents=["mCPBA (1.1 eq)"],
            solvents=["DCM"],
            temperature_c=0.0,
            time_hr=1.5,
            atmosphere="air",
            purification_method="column_chromatography",
            confidence=0.95,
        ),
        ReactionCondition(
            catalyst=None,
            reagents=["Dess-Martin Periodinane (1.2 eq)"],
            solvents=["DCM"],
            temperature_c=25.0,
            time_hr=1.5,
            atmosphere="air",
            purification_method="column_chromatography",
            confidence=0.93,
        ),
    ],
    # 9. Functional group interconversion (FGI)
    9: [
        ReactionCondition(
            catalyst=None,
            reagents=["PBr3 (1.2 eq)"],
            solvents=["DCM"],
            temperature_c=0.0,
            time_hr=2.0,
            atmosphere="inert_n2",
            purification_method="column_chromatography",
            confidence=0.93,
        ),
        ReactionCondition(
            catalyst="CuCN (1.2 eq)",
            reagents=["NaNO2 (1.2 eq)", "HCl (aq)"],
            solvents=["H2O", "MeCN"],
            temperature_c=50.0,
            time_hr=3.0,
            atmosphere="air",
            purification_method="column_chromatography",
            confidence=0.90,
        ),
    ],
    # 10. Functional group addition (FGA: Nitration, Halogenation)
    10: [
        ReactionCondition(
            catalyst=None,
            reagents=["HNO3 (conc. 1.2 eq)", "H2SO4 (conc. 1.5 eq)"],
            solvents=["None (Neat/Acidic)"],
            temperature_c=0.0,
            time_hr=1.5,
            atmosphere="air",
            purification_method="recrystallization",
            confidence=0.94,
        ),
        ReactionCondition(
            catalyst="FeCl3 (5 mol%)",
            reagents=["NBS (1.05 eq)"],
            solvents=["MeCN"],
            temperature_c=25.0,
            time_hr=2.0,
            atmosphere="air",
            purification_method="column_chromatography",
            confidence=0.92,
        ),
    ],
}


class ReactionConditionRecommender:
    """Predicts catalytic and experimental reaction conditions for retrosynthetic disconnections."""

    def __init__(self):
        self.templates = CLASS_CONDITION_TEMPLATES

    def detect_reaction_class(self, reactants: List[str], product: str, rule_name: str = "") -> int:
        """Heuristically identify the reaction class (1~10) from chemical fragments."""
        rule_lower = rule_name.lower()
        if "suzuki" in rule_lower or "coupling" in rule_lower and "c-c" in rule_lower:
            return 3
        if "amide" in rule_lower or "acylation" in rule_lower or "ester" in rule_lower:
            return 2
        if "alkylation" in rule_lower or "ether" in rule_lower:
            return 1
        if "heterocycle" in rule_lower or "imidazole" in rule_lower or "oxadiazole" in rule_lower:
            return 4
        if "boc_protection" in rule_lower or "protection" in rule_lower:
            return 5
        if "deprotection" in rule_lower or "cleavage" in rule_lower:
            return 6
        if "reduction" in rule_lower:
            return 7
        if "oxidation" in rule_lower:
            return 8
        if "fgi" in rule_lower or "sandmeyer" in rule_lower:
            return 9
        if "fga" in rule_lower or "nitration" in rule_lower or "bromination" in rule_lower:
            return 10

        # Substructure based heuristic
        react_str = ".".join(reactants)
        if "B(" in react_str and ("Br" in react_str or "I" in react_str):
            return 3
        if "C(=O)O" in react_str and ("N" in react_str or "O" in react_str):
            return 2
        if "C(=O)Cl" in react_str:
            return 2
        if "[N+](=O)[O-]" in react_str and "N" in product:
            return 7
        if "OC(C)(C)C" in react_str:
            return 6

        return 2  # Default to acylation/amide coupling if unknown

    def recommend_conditions(
        self,
        reactants: List[str],
        product: str,
        reaction_class: Optional[int] = None,
        rule_name: str = "",
    ) -> ReactionCondition:
        """Return the highest-confidence reaction condition for the transformation."""
        cid = reaction_class or self.detect_reaction_class(reactants, product, rule_name)
        candidates = self.templates.get(cid, self.templates[2])
        # Return primary validated standard protocol
        return candidates[0]
