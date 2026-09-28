"""Synthetic Accessibility Score (SAScore) calculation based on Ertl & Schuffenhauer.

Estimates ease of molecular synthesis (1 = very easy to synthesize, 10 = extremely difficult).
Uses RDKit RDContribDir SA_Score when available, with a robust heuristic fallback.
"""

import math
import os
import sys
from typing import Union

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

_SASCORE_MODULE = None


def _init_sascorer():
    """Lazily load sascorer from RDKit RDContribDir."""
    global _SASCORE_MODULE
    if _SASCORE_MODULE is not None:
        return _SASCORE_MODULE

    try:
        from rdkit.Chem import RDConfig

        sa_score_dir = os.path.join(RDConfig.RDContribDir, "SA_Score")
        if os.path.exists(sa_score_dir) and sa_score_dir not in sys.path:
            sys.path.append(sa_score_dir)
        import sascorer  # type: ignore

        _SASCORE_MODULE = sascorer
        return _SASCORE_MODULE
    except Exception:
        _SASCORE_MODULE = False
        return False


def _heuristic_sa_score(mol: Chem.Mol) -> float:
    """Fallback heuristic calculation of SAScore based on molecular complexity."""
    num_atoms = mol.GetNumHeavyAtoms()
    num_rings = rdMolDescriptors.CalcNumRings(mol)
    num_rotatable = Descriptors.NumRotatableBonds(mol)
    num_chiral = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    num_spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    num_bridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)

    # Complexity penalty
    penalty = (
        0.05 * num_atoms
        + 0.25 * num_rings
        + 0.10 * num_rotatable
        + 0.50 * num_chiral
        + 0.80 * num_spiro
        + 0.90 * num_bridgehead
    )
    score = 1.0 + math.log1p(max(0.0, penalty))
    return float(max(1.0, min(10.0, score)))


def calculate_sa_score(mol_or_smiles: Union[str, Chem.Mol]) -> float:
    """Calculate Synthetic Accessibility Score (SAScore) for a molecule.

    Args:
        mol_or_smiles: SMILES string or RDKit Mol object.

    Returns:
        float score between 1.0 (very easy) and 10.0 (very difficult).
    """
    if isinstance(mol_or_smiles, str):
        mol = Chem.MolFromSmiles(mol_or_smiles)
        if mol is None:
            return 10.0  # Invalid molecules get worst score
    else:
        mol = mol_or_smiles

    if mol is None or mol.GetNumHeavyAtoms() == 0:
        return 10.0

    scorer = _init_sascorer()
    if scorer:
        try:
            return float(scorer.calculateScore(mol))
        except Exception:
            return _heuristic_sa_score(mol)
    return _heuristic_sa_score(mol)


def is_synthetically_accessible(
    mol_or_smiles: Union[str, Chem.Mol], threshold: float = 3.5
) -> bool:
    """Check whether a molecule is synthetically accessible (SAScore <= threshold).

    Args:
        mol_or_smiles: SMILES string or RDKit Mol object.
        threshold: Maximum allowed SAScore (default: 3.5, medicinal chemistry standard).

    Returns:
        True if SAScore <= threshold, False otherwise.
    """
    return calculate_sa_score(mol_or_smiles) <= threshold
