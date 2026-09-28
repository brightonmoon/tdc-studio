"""Explainable AI (XAI) utilities for Drug-Target Interaction (DTI) Contact Maps and PyMOL integration."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# Standard 20 canonical amino acid single-letter to 3-letter code mapping
AA_1TO3 = {
    "A": "Ala", "R": "Arg", "N": "Asn", "D": "Asp", "C": "Cys",
    "E": "Glu", "Q": "Gln", "G": "Gly", "H": "His", "I": "Ile",
    "L": "Leu", "K": "Lys", "M": "Met", "F": "Phe", "P": "Pro",
    "S": "Ser", "T": "Thr", "W": "Trp", "Y": "Tyr", "V": "Val",
    "U": "Sec", "O": "Pyl",
}


def extract_top_contact_residues(
    contact_map: np.ndarray,
    target_seq: str,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """Aggregate 2D contact map to identify Top-K high-intensity target binding residues.

    Args:
        contact_map : 2D numpy array [L_drug, L_target] of cross-attention weights.
        target_seq  : Raw amino acid sequence string.
        top_k       : Number of top residues to return.

    Returns:
        List of dicts with keys: rank, index (1-based), residue_name, residue_code, score.
    """
    if contact_map.ndim != 2:
        return []

    l_drug, l_target = contact_map.shape
    actual_len = min(l_target, len(target_seq))
    if actual_len <= 0:
        return []

    # Slice contact map to actual sequence length (ignore trailing padding tokens)
    valid_map = contact_map[:, :actual_len]

    # Aggregate interaction intensity across all drug atoms/tokens
    residue_weights = np.sum(valid_map, axis=0)

    # Normalize scores to [0.0, 1.0] for intuitive interpretation
    max_val = float(np.max(residue_weights)) if len(residue_weights) > 0 else 0.0
    if max_val > 1e-8:
        norm_scores = residue_weights / max_val
    else:
        norm_scores = residue_weights

    # Get descending sorted indices
    k = min(top_k, actual_len)
    top_indices = np.argsort(-residue_weights)[:k]

    results = []
    for rank, idx in enumerate(top_indices, start=1):
        idx_int = int(idx)
        res_char = target_seq[idx_int].upper() if idx_int < len(target_seq) else "X"
        res_3 = AA_1TO3.get(res_char, res_char)
        res_1based = idx_int + 1
        res_code = f"{res_3}{res_1based}"

        results.append({
            "rank": rank,
            "index": res_1based,
            "residue_name": res_char,
            "residue_code": res_code,
            "score": round(float(norm_scores[idx_int]), 4),
        })

    return results


def extract_top_contact_atoms(
    contact_map: np.ndarray,
    smiles: str,
    tokens: Optional[List[str]] = None,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """Aggregate 2D contact map to identify Top-K high-intensity drug contact atoms/tokens.

    Args:
        contact_map : 2D numpy array [L_drug, L_target].
        smiles      : SMILES string of drug.
        tokens      : Optional list of token/atom names (from ChemBERTa or RDKit).
        top_k       : Number of top contact atoms to return.

    Returns:
        List of dicts with keys: rank, atom_index, token, score.
    """
    if contact_map.ndim != 2:
        return []

    l_drug, _ = contact_map.shape
    if l_drug <= 0:
        return []

    atom_weights = np.sum(contact_map, axis=1)
    max_val = float(np.max(atom_weights)) if len(atom_weights) > 0 else 0.0
    if max_val > 1e-8:
        norm_scores = atom_weights / max_val
    else:
        norm_scores = atom_weights

    k = min(top_k, l_drug)
    top_indices = np.argsort(-atom_weights)[:k]

    results = []
    for rank, idx in enumerate(top_indices, start=1):
        idx_int = int(idx)
        tok_str = tokens[idx_int] if (tokens and idx_int < len(tokens)) else f"Token_{idx_int}"
        results.append({
            "rank": rank,
            "atom_index": idx_int,
            "token": tok_str,
            "score": round(float(norm_scores[idx_int]), 4),
        })

    return results


def generate_pymol_command(
    top_residues: List[Dict[str, Any]],
    selection_name: str = "binding_pocket",
) -> str:
    """Generate ready-to-run PyMOL script snippet for structural pocket visualization.

    Example output:
        "select binding_pocket, resi 45+89+104+312; show sticks, binding_pocket; color magenta, binding_pocket; zoom binding_pocket, 6.0;"
    """
    if not top_residues:
        return ""

    resi_nums = [str(r["index"]) for r in top_residues if "index" in r]
    if not resi_nums:
        return ""

    resi_str = "+".join(resi_nums)
    return (
        f"select {selection_name}, resi {resi_str}; "
        f"show sticks, {selection_name}; "
        f"color magenta, {selection_name}; "
        f"zoom {selection_name}, 6.0;"
    )


def compute_affinity_consistency_score(
    pkd: float,
    pki: float,
    pic50: float,
) -> Tuple[float, str]:
    """Compute the biochemical Affinity Consistency Score (ACS) between Kd, Ki, and IC50.

    Principles:
    1. Competitive inhibitor expectation: Ki ~= Kd (same active site binding).
    2. Cheng-Prusoff relation: IC50 = Ki * (1 + [S]/Km) >= Ki  ==>  pIC50 <= pKi.
    3. Scale consistency: Inherent variance across log-affinity values.

    Returns:
        Tuple of (consistency_score [0.0, 100.0], tier ['High', 'Moderate', 'Review Required']).
    """
    vals = np.array([pkd, pki, pic50], dtype=np.float64)

    # 1. Variance penalty (dispersion)
    std_val = float(np.std(vals))
    dispersion_factor = np.exp(-std_val / 1.0)

    # 2. Physics / Cheng-Prusoff constraint violation penalty
    # In enzymatic assays: IC50 >= Ki  -->  pIC50 <= pKi + margin (0.3 log order)
    physics_penalty = 1.0
    if pic50 > pki + 0.3:
        excess = pic50 - (pki + 0.3)
        physics_penalty = float(np.exp(-excess / 0.5))

    # 3. Overall composite score (0 to 100)
    score = round(float(100.0 * dispersion_factor * physics_penalty), 1)
    score = min(max(score, 0.0), 100.0)

    # Tier assignment
    if score >= 75.0:
        tier = "High"
    elif score >= 50.0:
        tier = "Moderate"
    else:
        tier = "Review Required"

    return score, tier
