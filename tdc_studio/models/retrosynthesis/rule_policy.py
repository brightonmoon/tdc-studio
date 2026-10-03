"""Rule-based retrosynthetic disconnection policy using RDKit chemical reactions."""

import logging
from typing import Any, Dict, List, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import rdChemReactions

from tdc_studio.core.registry import MODELS
from tdc_studio.models.retrosynthesis.base import BaseRetroModel

logger = logging.getLogger("tdc_studio.models.retrosynthesis.rule_policy")

# 10 Standard Organic Reaction Retro-SMARTS Templates
# Formatted as: [Product Pattern] >> [Reactants Pattern]
DEFAULT_RETRO_RULES: List[Dict[str, Any]] = [
    {
        "id": "amide_coupling",
        "name": "Amide Coupling / Acylation",
        "class_id": 2,
        "smarts": "[C:1](=[O:2])[N:3]>>[C:1](=[O:2])O.[N:3]",
        "prior": 0.95,
    },
    {
        "id": "esterification",
        "name": "Ester Formation / Acylation",
        "class_id": 2,
        "smarts": "[C:1](=[O:2])[O:3][C;!$(C=O):4]>>[C:1](=[O:2])O.[O:3][C:4]",
        "prior": 0.88,
    },
    {
        "id": "sulfonamide",
        "name": "Sulfonamide Formation",
        "class_id": 2,
        "smarts": "[c,C:1][S:2](=[O:3])(=[O:4])[N:5]>>[c,C:1][S:2](=[O:3])(=[O:4])Cl.[N:5]",
        "prior": 0.92,
    },
    {
        "id": "suzuki_coupling",
        "name": "Suzuki-Miyaura C-C Cross Coupling",
        "class_id": 3,
        "smarts": "[c:1]-[c:2]>>[c:1]Br.OB(O)[c:2]",
        "prior": 0.90,
    },
    {
        "id": "aryl_ether_alkylation",
        "name": "Aryl Ether Synthesis / Alkylation",
        "class_id": 1,
        "smarts": "[c:1][O:2][C;!$(C=O):3]>>[c:1]O.[C:3]Br",
        "prior": 0.85,
    },
    {
        "id": "amine_alkylation",
        "name": "Amine Alkylation",
        "class_id": 1,
        "smarts": "[C:1][N;H1,H2:2]>>[C:1]Br.[N:2]",
        "prior": 0.82,
    },
    {
        "id": "reductive_amination",
        "name": "Reductive Amination",
        "class_id": 7,
        "smarts": "[C:1][NH:2][C:3]>>[C:1]=O.[NH2:2][C:3]",
        "prior": 0.87,
    },
    {
        "id": "urea_formation",
        "name": "Urea Formation",
        "class_id": 2,
        "smarts": "[N:1][C:2](=[O:3])[N:4]>>[N:1].[N:4]",
        "prior": 0.84,
    },
    {
        "id": "nitro_reduction",
        "name": "Aromatic Nitro Reduction",
        "class_id": 7,
        "smarts": "[c:1][NH2:2]>>[c:1][N+](=O)[O-]",
        "prior": 0.80,
    },
    {
        "id": "nitrile_reduction",
        "name": "Nitrile Reduction to Primary Amine",
        "class_id": 7,
        "smarts": "[C:1][CH2:2][NH2:3]>>[C:1]C#N",
        "prior": 0.78,
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
                            candidates.append((sorted_reactants, rule["prior"], rule["name"]))
            except Exception as exc:
                logger.debug(f"Rule {rule['id']} execution error: {exc}")

        # Sort candidates by prior score descending
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [(c[0], c[1]) for c in candidates[:top_k]]
