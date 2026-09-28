"""Bioisostere (Biological Isostere) Recommendation Engine for Defect Mitigation.

Applies codified medicinal chemistry transformation rules to replace offending toxicophores
and liabilities while preserving the active molecular scaffold.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from rdkit import Chem
from rdkit.Chem import AllChem


@dataclass
class BioisostereTransformation:
    """Transformation rule definition."""
    name: str
    target_smarts: str
    replacement_smiles: str
    liability_addressed: str
    rationale: str


# Codified medicinal chemistry bioisostere rules
BIOISOSTERE_RULES: List[BioisostereTransformation] = [
    # 1. Carboxylic acid liabilities (DILI, acyl glucuronide reactivity, poor permeability)
    BioisostereTransformation(
        name="cooh_to_tetrazole",
        target_smarts="[CX3](=O)[OX2H1,OX1-]",
        replacement_smiles="c1nnn[nH]1",
        liability_addressed="DILI & Glucuronidation Liability",
        rationale="Tetrazole is an established bioisostere of carboxylic acid with similar pKa (~4.5~4.9) but resistant to metabolic glucuronidation.",
    ),
    BioisostereTransformation(
        name="cooh_to_methylsulfonamide",
        target_smarts="[CX3](=O)[OX2H1,OX1-]",
        replacement_smiles="NS(=O)(=O)C",
        liability_addressed="Permeability & Metabolic Stability",
        rationale="Acyl/alkyl sulfonamide mimics acid hydrogen bonding while significantly improving membrane permeability.",
    ),

    # 2. Basic aliphatic amine liabilities (hERG potassium channel blockade)
    BioisostereTransformation(
        name="amine_to_morpholine",
        target_smarts="[NX3;H2,H1,H0;!$(NC=O)]",
        replacement_smiles="N1CCOCC1",
        liability_addressed="hERG Cardiotoxicity Blocker",
        rationale="Morpholine oxygen lowers basic amine pKa from ~10.0 to ~8.0, dramatically reducing hERG pore cavity electrostatic affinity.",
    ),
    BioisostereTransformation(
        name="amine_to_fluoroethylamine",
        target_smarts="[NX3;H2,H1,H0;!$(NC=O)]",
        replacement_smiles="NCC(F)F",
        liability_addressed="hERG Cardiotoxicity Blocker",
        rationale="Electronegative fluorines inductively attenuate amine basicity, attenuating cardiotoxic liability without breaking hydrogen bonding.",
    ),
    BioisostereTransformation(
        name="amine_to_amide",
        target_smarts="[NX3;H2,H1,H0;!$(NC=O)]",
        replacement_smiles="NC(=O)C",
        liability_addressed="hERG Cardiotoxicity Blocker",
        rationale="Converting amine to neutral amide eliminates cationic charge at pH 7.4 entirely.",
    ),

    # 3. Nitro group liabilities (AMES Mutagenicity & reactive metabolites)
    BioisostereTransformation(
        name="nitro_to_cyano",
        target_smarts="[N+](=O)[O-]",
        replacement_smiles="C#N",
        liability_addressed="AMES Genotoxicity & Mutagenicity",
        rationale="Cyano group retains strong electron-withdrawing character without undergoing bacterial/hepatic nitroreduction to toxic hydroxylamines.",
    ),
    BioisostereTransformation(
        name="nitro_to_trifluoromethyl",
        target_smarts="[N+](=O)[O-]",
        replacement_smiles="C(F)(F)F",
        liability_addressed="AMES Genotoxicity & Mutagenicity",
        rationale="Trifluoromethyl retains steric bulk and electron deficiency while eliminating mutagenic electrophilicity.",
    ),

    # 4. Labile ester liabilities (Rapid clearance & short half-life)
    BioisostereTransformation(
        name="ester_to_amide",
        target_smarts="[#6]C(=O)O[#6]",
        replacement_smiles="C(=O)N",
        liability_addressed="Rapid Hydrolysis & Short Half-Life",
        rationale="Amides are substantially more resistant to plasma and hepatic carboxylesterases than esters.",
    ),
    BioisostereTransformation(
        name="ester_to_oxadiazole",
        target_smarts="[#6]C(=O)O[#6]",
        replacement_smiles="c1ncno1",
        liability_addressed="Hydrolytic Clearance Liability",
        rationale="1,2,4-Oxadiazole heterocyclic ring is a classic non-hydrolyzable bioisostere of ester groups.",
    ),

    # 5. Aromatic metabolic soft spots (CYP450 rapid clearance)
    BioisostereTransformation(
        name="aryl_h_to_fluoro",
        target_smarts="[cH1:1]",
        replacement_smiles="[c:1]F",
        liability_addressed="Rapid CYP450 Hepatic Clearance",
        rationale="Fluorine substitution at metabolic soft spots blocks aromatic oxidation (CYP epoxidation) due to high C-F bond dissociation energy.",
    ),
]


class BioisostereRecommender:
    """Suggests bioisostere replacements for molecules with predicted ADMET liabilities."""

    def __init__(self, rules: Optional[List[BioisostereTransformation]] = None):
        self.rules = rules or BIOISOSTERE_RULES
        self._compiled_patterns = []
        for r in self.rules:
            pat = Chem.MolFromSmarts(r.target_smarts)
            rep = Chem.MolFromSmiles(r.replacement_smiles)
            if pat is not None and rep is not None:
                self._compiled_patterns.append((r, pat, rep))

    def recommend(
        self,
        smiles: str,
        liability_focus: Optional[str] = None,
        max_suggestions: int = 5,
    ) -> List[Dict[str, Any]]:
        """Generate bioisosteric modifications that resolve specific liabilities.

        Args:
            smiles: Input compound SMILES.
            liability_focus: Optional string filter ('herg', 'ames', 'dili', 'clearance').
            max_suggestions: Maximum number of candidate analogues to return.

        Returns:
            List of suggested modifications with modified SMILES and medicinal chemistry rationale.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []

        suggestions: List[Dict[str, Any]] = []
        seen_smiles = {Chem.MolToSmiles(mol)}

        for rule, target_pat, rep_mol in self._compiled_patterns:
            if liability_focus and liability_focus.lower() not in rule.liability_addressed.lower():
                continue

            if mol.HasSubstructMatch(target_pat):
                try:
                    # Replace all matching substructures
                    mod_mols = AllChem.ReplaceSubstructs(mol, target_pat, rep_mol, replaceAll=False)
                    for mod in mod_mols:
                        try:
                            Chem.SanitizeMol(mod)
                            mod_s = Chem.MolToSmiles(mod)
                            if mod_s not in seen_smiles:
                                seen_smiles.add(mod_s)
                                suggestions.append({
                                    "transformation_name": rule.name,
                                    "liability_addressed": rule.liability_addressed,
                                    "original_smiles": smiles,
                                    "modified_smiles": mod_s,
                                    "rationale": rule.rationale,
                                })
                                if len(suggestions) >= max_suggestions:
                                    return suggestions
                        except Exception:
                            continue
                except Exception:
                    continue

        return suggestions
