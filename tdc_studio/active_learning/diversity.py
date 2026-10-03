"""Chemical diversity filtering and scaffold exploration utilities.

Implements MaxMin Diversity selection based on Morgan Fingerprint Tanimoto distance,
ensuring that Top-K recommendations span structurally distinct chemical clusters
rather than picking 10 trivial analogues of the same core scaffold.
"""

from typing import Any, List, Optional

import numpy as np

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem

    _RDKIT_AVAILABLE = True
except ImportError:
    _RDKIT_AVAILABLE = False


def _get_fingerprint(smiles: str, radius: int = 2, n_bits: int = 1024) -> Optional[Any]:
    """Compute RDKit Morgan fingerprint bit vector from SMILES."""
    if not _RDKIT_AVAILABLE:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    except Exception:
        return None


def calculate_tanimoto_similarity(smiles1: str, smiles2: str) -> float:
    """Calculate Tanimoto similarity between two SMILES strings using Morgan fingerprints.

    Falls back to character n-gram Jaccard similarity if RDKit is not installed.
    """
    if smiles1 == smiles2:
        return 1.0

    if _RDKIT_AVAILABLE:
        fp1 = _get_fingerprint(smiles1)
        fp2 = _get_fingerprint(smiles2)
        if fp1 is not None and fp2 is not None:
            return float(DataStructs.TanimotoSimilarity(fp1, fp2))

    # Fallback: 3-character n-gram Jaccard similarity
    n = 3
    set1 = set([smiles1[i : i + n] for i in range(max(1, len(smiles1) - n + 1))])
    set2 = set([smiles2[i : i + n] for i in range(max(1, len(smiles2) - n + 1))])
    union_len = len(set1 | set2)
    if union_len == 0:
        return 0.0
    return len(set1 & set2) / union_len


def maxmin_diversity_picker(
    smiles_list: List[str],
    acquisition_scores: List[float],
    top_k: int = 10,
    min_distance: float = 0.35,
    tradeoff_lambda: float = 0.5,
) -> List[int]:
    """Select Top-K candidates using greedy MaxMin diversity combined with acquisition score.

    Objective at step k:
        argmax_{i in Candidates} [ (1 - lambda) * norm_score(i) + lambda * min_{j in Selected} Distance(i, j) ]
    where Distance = 1.0 - TanimotoSimilarity.

    Args:
        smiles_list: Candidate molecule SMILES.
        acquisition_scores: Acquisition function values (EI, UCB, etc.).
        top_k: Number of diverse candidates to pick.
        min_distance: Soft minimum distance constraint (1.0 - Tanimoto).
        tradeoff_lambda: Weight balancing acquisition score vs chemical diversity (0.0 to 1.0).

    Returns:
        List of selected candidate indices in order of selection.
    """
    n = len(smiles_list)
    if n == 0:
        return []
    if n <= top_k:
        return sorted(range(n), key=lambda i: acquisition_scores[i], reverse=True)

    scores = np.asarray(acquisition_scores, dtype=np.float64)
    min_sc, max_sc = float(np.min(scores)), float(np.max(scores))
    if max_sc > min_sc:
        norm_scores = (scores - min_sc) / (max_sc - min_sc)
    else:
        norm_scores = np.ones_like(scores)

    # Precompute or lazily compute fingerprints
    fps = [_get_fingerprint(s) for s in smiles_list] if _RDKIT_AVAILABLE else None

    # Step 1: Pick the candidate with the highest raw acquisition score
    best_first = int(np.argmax(scores))
    selected: List[int] = [best_first]
    remaining = set(range(n)) - {best_first}

    def pairwise_dist(idx_a: int, idx_b: int) -> float:
        if fps and fps[idx_a] is not None and fps[idx_b] is not None:
            sim = float(DataStructs.TanimotoSimilarity(fps[idx_a], fps[idx_b]))
        else:
            sim = calculate_tanimoto_similarity(smiles_list[idx_a], smiles_list[idx_b])
        return max(0.0, 1.0 - sim)

    # Step 2: Iteratively select remaining molecules with MaxMin criterion
    while len(selected) < top_k and remaining:
        best_candidate = -1
        best_val = -float("inf")

        for cand_idx in remaining:
            # Min distance to any already selected candidate
            min_dist = min(pairwise_dist(cand_idx, sel_idx) for sel_idx in selected)

            # Combined score
            comb_val = (1.0 - tradeoff_lambda) * norm_scores[cand_idx] + tradeoff_lambda * min_dist
            if min_dist < min_distance:
                comb_val *= 0.5  # soft penalty for being too close to an existing pick

            if comb_val > best_val:
                best_val = comb_val
                best_candidate = cand_idx

        if best_candidate != -1:
            selected.append(best_candidate)
            remaining.remove(best_candidate)
        else:
            break

    return selected
