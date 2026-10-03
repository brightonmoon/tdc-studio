"""Rule-based retrosynthetic disconnection policy using RDKit chemical reactions."""

import logging
from typing import Any, Dict, List, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import rdChemReactions

from tdc_studio.core.registry import MODELS
from tdc_studio.models.retrosynthesis.base import BaseRetroModel

logger = logging.getLogger("tdc_studio.models.retrosynthesis.rule_policy")

# Comprehensive Standard Organic Reaction Retro-SMARTS Templates (USPTO-50K 10 Classes)
# Formatted as: [Product Pattern] >> [Reactants Pattern]
DEFAULT_RETRO_RULES: List[Dict[str, Any]] = [
    # --- Class 1: Heteroatom alkylation and arylation ---
    {
        "id": "aryl_ether_alkylation",
        "name": "Aryl Ether Synthesis / Alkylation",
        "class_id": 1,
        "smarts": "[c:1][O:2][C:3]>>[c:1][OH:2].[C:3]Br",
        "prior": 0.94,
    },
    {
        "id": "alkyl_ether_alkylation",
        "name": "Alkyl Ether Synthesis",
        "class_id": 1,
        "smarts": "[C:1][O:2][C:3]>>[C:1][OH:2].[C:3]Br",
        "prior": 0.91,
    },
    {
        "id": "amine_alkylation",
        "name": "Amine Alkylation (Br)",
        "class_id": 1,
        "smarts": "[C:1][NH:2][C:3]>>[C:1]Br.[NH2:2][C:3]",
        "prior": 0.95,
    },
    {
        "id": "amine_alkylation_cl",
        "name": "Amine Alkylation (Cl)",
        "class_id": 1,
        "smarts": "[C:1][NH:2][C:3]>>[C:1]Cl.[NH2:2][C:3]",
        "prior": 0.90,
    },
    {
        "id": "tert_amine_alkylation_i",
        "name": "Tert-Amine Alkylation (I)",
        "class_id": 1,
        "smarts": "[C:1][N:2]([C:3])[C:4]>>[C:1]I.[NH:2]([C:3])[C:4]",
        "prior": 0.94,
    },
    {
        "id": "aniline_n_alkylation",
        "name": "Aniline N-Alkylation",
        "class_id": 1,
        "smarts": "[c:1][NH:2][C:3]>>[c:1]N.[C:3]Br",
        "prior": 0.96,
    },
    {
        "id": "aniline_n_arylation",
        "name": "Aniline N-Arylation",
        "class_id": 1,
        "smarts": "[c:1][NH:2][C:3]>>[c:1]Br.[C:3]N",
        "prior": 0.92,
    },
    {
        "id": "thioether_alkylation",
        "name": "Thioether Alkylation",
        "class_id": 1,
        "smarts": "[c,C:1][S:2][C:3]>>[c,C:1][SH:2].[C:3]I",
        "prior": 0.93,
    },
    {
        "id": "saponification_chloride",
        "name": "Ester Saponification / Chlorination",
        "class_id": 1,
        "smarts": "[C:1](=[O:2])[OH:3]>>[C:1](=[O:2])Cl.O",
        "prior": 0.85,
    },
    # --- Class 2: Acylation and related processes ---
    {
        "id": "amide_coupling",
        "name": "Amide Coupling (Acid + Amine)",
        "class_id": 2,
        "smarts": "[C:1](=[O:2])[N:3]>>[C:1](=[O:2])O.[N:3]",
        "prior": 0.96,
    },
    {
        "id": "amide_coupling_chloride",
        "name": "Amide Coupling (Chloride + Amine)",
        "class_id": 2,
        "smarts": "[C:1](=[O:2])[N:3]>>[C:1](=[O:2])Cl.[N:3]",
        "prior": 0.95,
    },
    {
        "id": "esterification",
        "name": "Ester Formation (Chloride + Alcohol)",
        "class_id": 2,
        "smarts": "[C:1](=[O:2])[O:3][C:4]>>[C:1](=[O:2])Cl.[OH:3][C:4]",
        "prior": 0.94,
    },
    {
        "id": "esterification_acid",
        "name": "Ester Formation (Acid + Alcohol)",
        "class_id": 2,
        "smarts": "[C:1](=[O:2])[O:3][C:4]>>[C:1](=[O:2])O.[OH:3][C:4]",
        "prior": 0.92,
    },
    {
        "id": "sulfonamide",
        "name": "Sulfonamide Formation",
        "class_id": 2,
        "smarts": "[c,C:1][S:2](=[O:3])(=[O:4])[N:5]>>[c,C:1][S:2](=[O:3])(=[O:4])Cl.[N:5]",
        "prior": 0.95,
    },
    {
        "id": "urea_formation",
        "name": "Urea Formation",
        "class_id": 2,
        "smarts": "[N:1][C:2](=[O:3])[N:4]>>[N:1].[N:4]",
        "prior": 0.84,
    },
    # --- Class 3: C-C bond formation ---
    {
        "id": "suzuki_coupling",
        "name": "Suzuki-Miyaura Biaryl Coupling",
        "class_id": 3,
        "smarts": "[c:1]-[c:2]>>[c:1]Br.OB(O)[c:2]",
        "prior": 0.96,
    },
    {
        "id": "suzuki_coupling_rev",
        "name": "Suzuki-Miyaura Biaryl Coupling (Inverted)",
        "class_id": 3,
        "smarts": "[c:1]-[c:2]>>OB(O)[c:1].[c:2]Br",
        "prior": 0.95,
    },
    {
        "id": "suzuki_benzyl",
        "name": "Suzuki Benzyl Coupling",
        "class_id": 3,
        "smarts": "[c:1][C:2][c:3]>>[c:1][C:2]Br.OB(O)[c:3]",
        "prior": 0.93,
    },
    {
        "id": "sonogashira_coupling_i",
        "name": "Sonogashira Coupling (Iodide)",
        "class_id": 3,
        "smarts": "[c:1]-[C:2]#[C:3]>>[c:1]I.C#[C:3]",
        "prior": 0.95,
    },
    {
        "id": "sonogashira_coupling_br",
        "name": "Sonogashira Coupling (Bromide)",
        "class_id": 3,
        "smarts": "[c:1]-[C:2]#[C:3]>>[c:1]Br.C#[C:3]",
        "prior": 0.92,
    },
    # --- Class 4: Heterocycle formation ---
    {
        "id": "benzimidazole_formation",
        "name": "Benzimidazole Formation",
        "class_id": 4,
        "smarts": "[c:1]1:[c:2]:[c:3]:[c:4]2:[n;H1:5]:[c:6]:[n:7]:[c:8]2:[c:9]1>>[c:1]1:[c:2]:[c:3]:[c:4]([NH2:5]):[c:8]([NH2:7]):[c:9]1.O=C[O:6]",
        "prior": 0.94,
    },
    {
        "id": "oxadiazole_formation",
        "name": "1,2,4-Oxadiazole Formation",
        "class_id": 4,
        "smarts": "[c:1]1[n:2][o:3][c:4][n:5]1>>[c:1](=[N:2])[NH2:5]O.[c:4](=O)Cl",
        "prior": 0.95,
    },
    {
        "id": "benzoxazole_formation",
        "name": "Benzoxazole Formation",
        "class_id": 4,
        "smarts": "[c:1]1:[c:2]:[c:3]:[c:4]2:[o:5]:[c:6]:[n:7]:[c:8]2:[c:9]1>>[c:1]1:[c:2]:[c:3]:[c:4]([OH:5]):[c:8]([NH2:7]):[c:9]1.O=C[O:6]",
        "prior": 0.93,
    },
    {
        "id": "benzothiazole_formation",
        "name": "Benzothiazole Formation",
        "class_id": 4,
        "smarts": "[c:1]1:[c:2]:[c:3]:[c:4]2:[s:5]:[c:6]:[n:7]:[c:8]2:[c:9]1>>[c:1]1:[c:2]:[c:3]:[c:4]([SH:5]):[c:8]([NH2:7]):[c:9]1.O=C[O:6]",
        "prior": 0.93,
    },
    {
        "id": "thiazole_formation",
        "name": "1,3-Thiazole Formation",
        "class_id": 4,
        "smarts": "[c:1]1[s:2][c:3][n:4][c:5]1>>[c:1](=[S:2])[NH2:4].[c:3](=O)[C:5]Br",
        "prior": 0.94,
    },
    # --- Class 5: Protections ---
    {
        "id": "boc_protection_ar",
        "name": "Boc Protection (Aromatic Amine)",
        "class_id": 5,
        "smarts": "[c:1][NH:2]C(=O)OC(C)(C)C>>[c:1][NH2:2].CC(C)(C)OC(=O)OC(=O)OC(C)(C)C",
        "prior": 0.97,
    },
    {
        "id": "boc_protection_sec",
        "name": "Boc Protection (Secondary Amine)",
        "class_id": 5,
        "smarts": "[C:1][N:2]([C:3])C(=O)OC(C)(C)C>>[C:1][NH:2][C:3].CC(C)(C)OC(=O)OC(=O)OC(C)(C)C",
        "prior": 0.96,
    },
    {
        "id": "silyl_protection",
        "name": "TBDMS Silyl Protection",
        "class_id": 5,
        "smarts": "[c:1][O:2][Si](C)(C)C(C)(C)C>>[c:1][OH:2].CC(C)(C)[Si](C)(C)Cl",
        "prior": 0.96,
    },
    {
        "id": "cbz_protection",
        "name": "Cbz Protection",
        "class_id": 5,
        "smarts": "[c:1][NH:2]C(=O)OCc2ccccc2>>[c:1][NH2:2].c1ccc(COC(=O)Cl)cc1",
        "prior": 0.95,
    },
    {
        "id": "acetyl_protection_phenol",
        "name": "Acetyl Protection (Phenol)",
        "class_id": 5,
        "smarts": "[c:1][O:2]C(=O)C>>[c:1][OH:2].CC(=O)Cl",
        "prior": 0.94,
    },
    {
        "id": "acetyl_protection_amine",
        "name": "Acetyl Protection (Amine)",
        "class_id": 5,
        "smarts": "[c:1][N:2](C)C(=O)C>>[c:1][NH:2]C.CC(=O)Cl",
        "prior": 0.94,
    },
    # --- Class 6: Deprotections ---
    {
        "id": "boc_deprotection_ar",
        "name": "Boc Deprotection (Aromatic Amine)",
        "class_id": 6,
        "smarts": "[c:1][NH2:2]>>[c:1][NH:2]C(=O)OC(C)(C)C",
        "prior": 0.96,
    },
    {
        "id": "boc_deprotection_sec",
        "name": "Boc Deprotection (Secondary Amine)",
        "class_id": 6,
        "smarts": "[C:1][NH:2][C:3]>>[C:1][N:2]([C:3])C(=O)OC(C)(C)C",
        "prior": 0.95,
    },
    {
        "id": "silyl_deprotection",
        "name": "Silyl Ether Deprotection",
        "class_id": 6,
        "smarts": "[c:1][OH:2]>>[c:1][O:2][Si](C)(C)C(C)(C)C",
        "prior": 0.96,
    },
    {
        "id": "ester_hydrolysis_deprotection",
        "name": "Ester Hydrolysis / Deprotection",
        "class_id": 6,
        "smarts": "[c:1]C(=O)O>>[c:1]C(=O)OCC",
        "prior": 0.95,
    },
    {
        "id": "acetyl_deprotection_phenol",
        "name": "Acetyl Cleavage (Phenol)",
        "class_id": 6,
        "smarts": "[c:1]O>>[c:1]OC(=O)C",
        "prior": 0.94,
    },
    {
        "id": "acetyl_deprotection_amine",
        "name": "Acetyl Cleavage (Amine)",
        "class_id": 6,
        "smarts": "[c:1][NH:2]C>>[c:1][N:2](C)C(=O)C",
        "prior": 0.94,
    },
    # --- Class 7: Reductions ---
    {
        "id": "nitro_reduction",
        "name": "Aromatic Nitro Reduction",
        "class_id": 7,
        "smarts": "[c:1][NH2:2]>>[c:1][N+](=O)[O-]",
        "prior": 0.97,
    },
    {
        "id": "aldehyde_reduction",
        "name": "Aldehyde Reduction to Primary Alcohol",
        "class_id": 7,
        "smarts": "[c:1][CH2:2][OH:3]>>[c:1][CH:2]=O",
        "prior": 0.96,
    },
    {
        "id": "ketone_reduction",
        "name": "Ketone Reduction to Secondary Alcohol",
        "class_id": 7,
        "smarts": "[C:1][CH:2]([OH:3])[C:4]>>[C:1][C:2](=O)[C:4]",
        "prior": 0.95,
    },
    {
        "id": "nitrile_reduction",
        "name": "Nitrile Reduction to Primary Amine",
        "class_id": 7,
        "smarts": "[c:1][CH2:2][CH2:3][NH2:4]>>[c:1][CH2:2][C:3]#N",
        "prior": 0.94,
    },
    {
        "id": "carboxylic_reduction",
        "name": "Carboxylic Acid Reduction",
        "class_id": 7,
        "smarts": "[c:1][CH2:2][CH2:3][OH:4]>>[c:1][CH2:2][C:3](=O)O",
        "prior": 0.93,
    },
    {
        "id": "reductive_amination",
        "name": "Reductive Amination",
        "class_id": 7,
        "smarts": "[C:1][NH:2][C:3]>>[C:1]=O.[NH2:2][C:3]",
        "prior": 0.87,
    },
    # --- Class 8: Oxidations ---
    {
        "id": "alcohol_to_aldehyde",
        "name": "Primary Alcohol Oxidation to Aldehyde",
        "class_id": 8,
        "smarts": "[c:1][CH:2]=O>>[c:1][CH2:2]O",
        "prior": 0.96,
    },
    {
        "id": "aldehyde_to_acid",
        "name": "Aldehyde Oxidation to Carboxylic Acid",
        "class_id": 8,
        "smarts": "[c:1][C:2](=O)O>>[c:1][CH:2]=O",
        "prior": 0.96,
    },
    {
        "id": "alcohol_to_ketone",
        "name": "Secondary Alcohol Oxidation to Ketone",
        "class_id": 8,
        "smarts": "[C:1][C:2](=O)[c:3]>>[C:1][CH:2](O)[c:3]",
        "prior": 0.95,
    },
    {
        "id": "sulfide_to_sulfoxide",
        "name": "Sulfide Oxidation to Sulfoxide",
        "class_id": 8,
        "smarts": "[c:1][S:2](=O)[C:3]>>[c:1][SH0:2][C:3]",
        "prior": 0.96,
    },
    {
        "id": "sulfoxide_to_sulfone",
        "name": "Sulfoxide Oxidation to Sulfone",
        "class_id": 8,
        "smarts": "[c:1][S:2](=O)(=O)[C:3]>>[c:1][S:2](=O)[C:3]",
        "prior": 0.95,
    },
    {
        "id": "pyridine_oxidation",
        "name": "Pyridine Alcohol Oxidation",
        "class_id": 8,
        "smarts": "[c:1][C:2](=O)O>>[c:1][CH2:2]O",
        "prior": 0.94,
    },
    # --- Class 9: Functional group interconversion (FGI) ---
    {
        "id": "fgi_aniline_to_chloride",
        "name": "Sandmeyer Chlorination",
        "class_id": 9,
        "smarts": "[c:1]Cl>>[c:1]N",
        "prior": 0.96,
    },
    {
        "id": "fgi_aniline_to_bromide",
        "name": "Sandmeyer Bromination",
        "class_id": 9,
        "smarts": "[c:1]Br>>[c:1]N",
        "prior": 0.96,
    },
    {
        "id": "fgi_bromide_to_nitrile",
        "name": "Aryl Cyanation",
        "class_id": 9,
        "smarts": "[c:1]C#N>>[c:1]Br",
        "prior": 0.95,
    },
    {
        "id": "fgi_alcohol_to_chloride",
        "name": "Alcohol Chlorination",
        "class_id": 9,
        "smarts": "[c:1][CH2:2]Cl>>[c:1][CH2:2]O",
        "prior": 0.95,
    },
    {
        "id": "fgi_alcohol_to_bromide",
        "name": "Alcohol Bromination",
        "class_id": 9,
        "smarts": "[c:1][CH2:2]Br>>[c:1][CH2:2]O",
        "prior": 0.95,
    },
    {
        "id": "fgi_bromide_to_nitrile_al",
        "name": "Alkyl Cyanation",
        "class_id": 9,
        "smarts": "[c:1][CH2:2]C#N>>[c:1][CH2:2]Br",
        "prior": 0.94,
    },
    # --- Class 10: Functional group addition (FGA) ---
    {
        "id": "fga_nitration",
        "name": "Electrophilic Aromatic Nitration",
        "class_id": 10,
        "smarts": "[c:1][N+](=O)[O-]>>[c:1]",
        "prior": 0.96,
    },
    {
        "id": "fga_bromination",
        "name": "Electrophilic Aromatic Bromination",
        "class_id": 10,
        "smarts": "[c:1]Br>>[c:1]",
        "prior": 0.95,
    },
    {
        "id": "fga_chlorination",
        "name": "Electrophilic Aromatic Chlorination",
        "class_id": 10,
        "smarts": "[c:1]Cl>>[c:1]",
        "prior": 0.94,
    },
    {
        "id": "fga_chlorosulfonation",
        "name": "Electrophilic Chlorosulfonation",
        "class_id": 10,
        "smarts": "[c:1]S(=O)(=O)Cl>>[c:1]",
        "prior": 0.94,
    },
]



@MODELS.register("rule_retro_policy")
class RuleRetroPolicy(BaseRetroModel):
    """Deterministic, fast rule-based retrosynthetic policy engine."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.rules = []
        rule_list = self.config.get("rules", DEFAULT_RETRO_RULES)

        for r_cfg in rule_list:
            try:
                rxn = rdChemReactions.ReactionFromSmarts(r_cfg["smarts"])
                if rxn is not None:
                    self.rules.append(
                        {
                            "id": r_cfg["id"],
                            "name": r_cfg["name"],
                            "class_id": r_cfg.get("class_id", 0),
                            "rxn": rxn,
                            "prior": float(r_cfg.get("prior", 0.8)),
                        }
                    )
            except Exception as e:
                logger.warning(f"Failed to compile reaction rule '{r_cfg.get('id')}': {e}")

    def predict_reactants(
        self,
        product_smiles: str,
        top_k: int = 5,
        reaction_type: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """Apply all compiled reaction rules to the product and return top-k precursors."""
        mol = Chem.MolFromSmiles(product_smiles)
        if mol is None:
            return []

        candidates: List[Tuple[str, float, str]] = []
        seen_reactants = set()

        for rule in self.rules:
            # If specific reaction_type is requested, skip non-matching rules
            if reaction_type is not None and rule["class_id"] != reaction_type:
                continue

            try:
                outcomes = rule["rxn"].RunReactants((mol,))
                for product_tuple in outcomes:
                    frag_smiles = []
                    valid_outcome = True
                    for frag_mol in product_tuple:
                        try:
                            Chem.SanitizeMol(frag_mol)
                            smi = Chem.MolToSmiles(frag_mol, canonical=True)
                            if smi:
                                frag_smiles.append(smi)
                            else:
                                valid_outcome = False
                        except Exception:
                            valid_outcome = False
                            break

                    if valid_outcome and frag_smiles:
                        # Deterministic canonical ordering of reactant fragments
                        sorted_reactants = ".".join(sorted(frag_smiles))
                        if sorted_reactants not in seen_reactants:
                            seen_reactants.add(sorted_reactants)
                            candidates.append(
                                (sorted_reactants, rule["prior"], rule["name"])
                            )
            except Exception as exc:
                logger.debug(f"Rule {rule['id']} execution error: {exc}")

        # Sort candidates by prior score descending
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [(c[0], c[1]) for c in candidates[:top_k]]
