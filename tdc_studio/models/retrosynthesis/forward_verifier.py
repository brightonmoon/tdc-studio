"""Forward reaction verifier for Round-Trip validation (Reactants -> Product == Expected)."""

import logging
from typing import Any, Dict, List, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import rdChemReactions

from tdc_studio.core.registry import MODELS
from tdc_studio.data.retrosyn import remove_atom_mapping
from tdc_studio.models.retrosynthesis.base import BaseForwardModel

logger = logging.getLogger("tdc_studio.models.retrosynthesis.forward_verifier")

# Forward Reaction SMARTS for common organic coupling transformations
DEFAULT_FORWARD_RULES: List[Dict[str, Any]] = [
    {
        "id": "fwd_amide_coupling",
        "name": "Forward Amide Formation",
        "smarts": "[C:1](=[O:2])O.[N;H1,H2:3]>>[C:1](=[O:2])[N:3]",
        "confidence": 0.95,
    },
    {
        "id": "fwd_acyl_chloride_amide",
        "name": "Forward Acid Chloride Amide Formation",
        "smarts": "[C:1](=[O:2])Cl.[N;H1,H2:3]>>[C:1](=[O:2])[N:3]",
        "confidence": 0.98,
    },
    {
        "id": "fwd_esterification",
        "name": "Forward Esterification",
        "smarts": "[C:1](=[O:2])O.[O;H1:3][C:4]>>[C:1](=[O:2])[O:3][C:4]",
        "confidence": 0.90,
    },
    {
        "id": "fwd_sulfonamide",
        "name": "Forward Sulfonamide Formation",
        "smarts": "[c,C:1][S:2](=[O:3])(=[O:4])Cl.[N;H1,H2:5]>>[c,C:1][S:2](=[O:3])(=[O:4])[N:5]",
        "confidence": 0.94,
    },
    {
        "id": "fwd_suzuki_coupling",
        "name": "Forward Suzuki-Miyaura Coupling",
        "smarts": "[c:1][Br,I].[c:2]B(O)O>>[c:1]-[c:2]",
        "confidence": 0.92,
    },
    {
        "id": "fwd_aryl_ether",
        "name": "Forward Williamson Ether Synthesis",
        "smarts": "[c:1]O.[C:2][Br,I]>>[c:1][O:2]",
        "confidence": 0.88,
    },
    {
        "id": "fwd_reductive_amination",
        "name": "Forward Reductive Amination",
        "smarts": "[C:1]=O.[N;H1,H2:2][C:3]>>[C:1][NH:2][C:3]",
        "confidence": 0.89,
    },
]


@MODELS.register("forward_verifier")
class ForwardVerifier(BaseForwardModel):
    """Verifies retrosynthetic steps by simulating the forward chemical reaction."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.rules = []
        rule_list = self.config.get("rules", DEFAULT_FORWARD_RULES)

        for r in rule_list:
            try:
                rxn = rdChemReactions.ReactionFromSmarts(r["smarts"])
                if rxn is not None:
                    self.rules.append(
                        {
                            "id": r["id"],
                            "name": r["name"],
                            "rxn": rxn,
                            "confidence": float(r.get("confidence", 0.9)),
                        }
                    )
            except Exception as e:
                logger.warning(f"Failed to compile forward rule '{r.get('id')}': {e}")

    def predict_product(
        self,
        reactants_smiles: str,
        top_k: int = 1,
    ) -> List[Tuple[str, float]]:
        """Predict possible forward products given reactant fragments."""
        frags = [f.strip() for f in reactants_smiles.split(".") if f.strip()]
        mol_frags = [Chem.MolFromSmiles(f) for f in frags]
        if any(m is None for m in mol_frags):
            return []

        candidates = []
        seen = set()

        for rule in self.rules:
            try:
                rxn: rdChemReactions.ChemicalReaction = rule["rxn"]
                num_reactants = rxn.GetNumReactantTemplates()
                if num_reactants == len(mol_frags):
                    outcomes = rxn.RunReactants(tuple(mol_frags))
                    for outcome in outcomes:
                        for prod_mol in outcome:
                            try:
                                Chem.SanitizeMol(prod_mol)
                                canon_smi = Chem.MolToSmiles(prod_mol, canonical=True)
                                if canon_smi and canon_smi not in seen:
                                    seen.add(canon_smi)
                                    candidates.append((canon_smi, rule["confidence"]))
                            except Exception:
                                pass
                elif num_reactants == 2 and len(mol_frags) == 2:
                    # Try reverse argument order (A + B vs B + A)
                    outcomes = rxn.RunReactants((mol_frags[1], mol_frags[0]))
                    for outcome in outcomes:
                        for prod_mol in outcome:
                            try:
                                Chem.SanitizeMol(prod_mol)
                                canon_smi = Chem.MolToSmiles(prod_mol, canonical=True)
                                if canon_smi and canon_smi not in seen:
                                    seen.add(canon_smi)
                                    candidates.append((canon_smi, rule["confidence"]))
                            except Exception:
                                pass
            except Exception as exc:
                logger.debug(f"Forward rule execution error: {exc}")

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]

    def verify_reaction(
        self,
        reactants_smiles: str,
        expected_product_smiles: str,
    ) -> Tuple[bool, float]:
        """Verify whether proposed reactants can indeed form the expected target product.

        Returns:
            (is_round_trip_valid: bool, verification_score: float)
        """
        clean_expected = remove_atom_mapping(expected_product_smiles)
        mol_exp = Chem.MolFromSmiles(clean_expected)
        canon_expected = (
            Chem.MolToSmiles(mol_exp, canonical=True)
            if mol_exp is not None
            else clean_expected
        )

        preds = self.predict_product(reactants_smiles, top_k=5)

        for prod_smi, conf in preds:
            clean_pred = remove_atom_mapping(prod_smi)
            mol_pred = Chem.MolFromSmiles(clean_pred)
            canon_pred = (
                Chem.MolToSmiles(mol_pred, canonical=True)
                if mol_pred is not None
                else clean_pred
            )
            if canon_pred == canon_expected:
                return True, conf

        return False, 0.0
