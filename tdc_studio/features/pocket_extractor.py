"""Binding Pocket Extraction and Domain-Centric Pocket Window Extractor.

Addresses the 1,024 AA truncation limitation:
- For long proteins (>1,000 AA) like EGFR (1,210 AA) or Titin, naive truncation cuts off
  critical C-terminal or middle catalytic domains.
- BindingPocketExtractor identifies catalytic motifs, conserved hydrophobic pocket cores,
  or AlphaFold PDB structural cavity coordinates, extracting a concentrated active pocket window
  (default 128-256 AA) with exact 1-to-1 residue coordinate mapping.
"""

import logging
import os
import re
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("tdc_studio.features.pocket")

# Canonical kinase & catalytic domain signature motifs (PROSITE-style regex)
_KNOWN_CATALYTIC_MOTIFS = [
    # Protein kinase ATP-binding region: Gly-rich loop [LIV]-G-{P}-G-{P}-[FYWMGSTNH]-[SGA]-{PW}-[LIVCAT]-{PE}-x-[LIVM]-[GSTAPIV]
    re.compile(r"[LIV]G.G.[FYWMGSTNH][SGA].[LIVCAT].{0,2}[LIVM][GSTAPIV]"),
    # Protein kinase catalytic loop: [LIVMFYC]-[HY]-x-D-[LIVMFY]-K-x-x-N-[LIVMFYC]{3}
    re.compile(r"[LIVMFYC][HY].D[LIVMFY]K..N[LIVMFYC]{3}"),
    # Serine proteases, trypsin family: G-D-S-G-[GS]-[PE]
    re.compile(r"GDSG[GS][PE]"),
    # Cysteine proteases, papain family: C-x-[GS]-x-C-W
    re.compile(r"C.[GS].CW"),
    # Aspartic proteases active site: [LIVMFG]-[LIVM]-D-[TS]-G-[ST]-[ST]
    re.compile(r"[LIVMFG][LIVM]D[TS]G[ST]{2}"),
]


class BindingPocketExtractor:
    """Extracts binding pocket residues and domain-centric sub-sequences from proteins.

    Supports:
    1. Structure-based pocket detection from AlphaFold PDB files (high pLDDT + cavity density).
    2. Motif-based catalytic site detection (Kinases, Proteases, GPCR transmembrane bundles).
    3. Concentrated pocket window extraction with exact residue coordinate mapping.

    Args:
        cache_dir: Directory to cache downloaded AlphaFold PDB files.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir or "data/structures/alphafold")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_alphafold_pdb(self, uniprot_id: str) -> Optional[str]:
        """Fetch AlphaFold predicted structure PDB content for a UniProt ID."""
        clean_id = uniprot_id.strip().upper()
        cache_file = self.cache_dir / f"AF-{clean_id}-F1.pdb"
        if cache_file.is_file():
            try:
                return cache_file.read_text(encoding="utf-8")
            except Exception as e:
                logger.warning("Failed reading cached PDB %s: %s", cache_file, e)

        # AlphaFold v4 URL pattern
        url = f"https://alphafold.ebi.ac.uk/files/AF-{clean_id}-F1-model_v4.pdb"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "TDC-Studio/1.0"})
            with urllib.request.urlopen(req, timeout=10) as response:
                pdb_text = response.read().decode("utf-8")
                cache_file.write_text(pdb_text, encoding="utf-8")
                return pdb_text
        except Exception as exc:
            logger.debug("AlphaFold DB download skipped for '%s': %s", clean_id, exc)
            return None

    def find_motif_centers(self, seq: str) -> List[int]:
        """Find central residue indices of conserved catalytic/binding motifs."""
        centers = []
        for motif_re in _KNOWN_CATALYTIC_MOTIFS:
            for match in motif_re.finditer(seq):
                center = (match.start() + match.end()) // 2
                centers.append(center)
        return centers

    def extract_structure_cavity_indices(
        self, pdb_text: str, top_k: int = 128, min_plddt: float = 70.0
    ) -> List[int]:
        """Extract high-confidence surface cavity residue indices from PDB content.

        Reads CA (C-alpha) coordinates and B-factor (pLDDT metric in AlphaFold PDBs).
        Identifies spatial clusters of residues with pLDDT >= min_plddt.
        """
        ca_coords = []
        plddt_scores = []
        res_indices = []

        for line in pdb_text.splitlines():
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                try:
                    res_num = int(line[22:26].strip())
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    plddt = float(line[60:66].strip())

                    res_indices.append(res_num - 1)  # 0-indexed
                    ca_coords.append([x, y, z])
                    plddt_scores.append(plddt)
                except (ValueError, IndexError):
                    continue

        if not ca_coords:
            return []

        coords_arr = np.array(ca_coords, dtype=np.float32)
        plddt_arr = np.array(plddt_scores, dtype=np.float32)

        # Filter by high AlphaFold confidence (pLDDT >= 70)
        confident_mask = plddt_arr >= min_plddt
        if not np.any(confident_mask):
            confident_mask = np.ones(len(plddt_arr), dtype=bool)

        conf_coords = coords_arr[confident_mask]
        conf_indices = [res_indices[i] for i, m in enumerate(confident_mask) if m]

        # Compute local neighbor density within 10 Angstrom sphere
        # Pocket cavities typically have high neighbor packing in folded structures
        if len(conf_coords) <= top_k:
            return conf_indices

        diff = conf_coords[:, None, :] - conf_coords[None, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        # Neighbors within 10.0 Angstrom
        neighbor_counts = np.sum((dists < 10.0) & (dists > 0.1), axis=-1)

        # Select top residues with highest local structured density
        top_local_idx = np.argsort(-neighbor_counts)[:top_k]
        selected_res = [conf_indices[idx] for idx in sorted(top_local_idx)]
        return selected_res

    def extract_pocket_window(
        self,
        seq: str,
        max_window_len: int = 256,
        uniprot_id: Optional[str] = None,
        pdb_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Extract a continuous pocket-focused window of length <= max_window_len.

        Returns:
            Dict containing:
                "pocket_seq"      : Extracted amino acid sub-sequence string.
                "start_idx"       : 0-based start index in original sequence.
                "end_idx"         : 0-based end index in original sequence.
                "coordinate_map"  : List of original 1-based residue indices.
                "is_truncated"    : True if original sequence exceeded max_window_len.
                "method"          : "alphafold_cavity", "catalytic_motif", or "center_window".
        """
        seq_len = len(seq)
        if seq_len <= max_window_len:
            return {
                "pocket_seq": seq,
                "start_idx": 0,
                "end_idx": seq_len,
                "coordinate_map": list(range(1, seq_len + 1)),
                "is_truncated": False,
                "method": "full_sequence",
            }

        # Priority 1: Structure-based cavity centers from AlphaFold PDB
        structure_pdb = pdb_text
        if structure_pdb is None and uniprot_id:
            structure_pdb = self.fetch_alphafold_pdb(uniprot_id)

        target_center = None
        method = "center_window"

        if structure_pdb:
            cavity_indices = self.extract_structure_cavity_indices(structure_pdb, top_k=64)
            if cavity_indices:
                target_center = int(np.median(cavity_indices))
                method = "alphafold_cavity"

        # Priority 2: Conserved catalytic/kinase ATP-binding motifs
        if target_center is None:
            motif_centers = self.find_motif_centers(seq)
            if motif_centers:
                # Pick the most central or first major motif
                target_center = motif_centers[0]
                method = "catalytic_motif"

        # Priority 3: Fallback to sequence middle third
        if target_center is None:
            target_center = seq_len // 2
            method = "center_window"

        # Center window around target_center
        half = max_window_len // 2
        start_idx = max(0, target_center - half)
        end_idx = min(seq_len, start_idx + max_window_len)

        # Adjust start if end hit sequence boundary
        if end_idx - start_idx < max_window_len and start_idx > 0:
            start_idx = max(0, end_idx - max_window_len)

        pocket_seq = seq[start_idx:end_idx]
        coord_map = list(range(start_idx + 1, end_idx + 1))

        return {
            "pocket_seq": pocket_seq,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "coordinate_map": coord_map,
            "is_truncated": True,
            "method": method,
        }

    def extract_pocket_residues_from_p2rank(
        self,
        p2rank_csv: str,
        pocket_rank: int = 1,
    ) -> List[int]:
        """Extract 0-based residue indices for the top pocket from P2Rank prediction CSV."""
        pockets = parse_p2rank_predictions(p2rank_csv)
        for p in pockets:
            if p["rank"] == pocket_rank:
                return p["residue_indices_0based"]
        return pockets[0]["residue_indices_0based"] if pockets else []


def parse_p2rank_predictions(
    content_or_path: str,
) -> List[Dict[str, Any]]:
    """Parse P2Rank pocket prediction CSV output.

    P2Rank output format:
        name, rank, score, probability, center_x, center_y, center_z, residue_ids, surf_atom_ids
    Where residue_ids is space-separated strings (e.g. 'A_745 A_746 A_750 ...').

    Args:
        content_or_path: File path to *_predictions.csv or raw CSV text string.

    Returns:
        List of pocket dicts containing:
            "rank": int,
            "name": str,
            "score": float,
            "probability": float,
            "center": (x, y, z),
            "residue_indices_1based": List[int],
            "residue_indices_0based": List[int],
            "raw_residue_ids": List[str],
    """
    text = content_or_path
    if os.path.exists(content_or_path):
        with open(content_or_path, "r", encoding="utf-8") as f:
            text = f.read()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []

    # Detect header
    header_idx = -1
    for idx, line in enumerate(lines):
        if "name" in line.lower() and "rank" in line.lower() and "score" in line.lower():
            header_idx = idx
            break

    if header_idx == -1:
        return []

    headers = [h.strip().lower() for h in lines[header_idx].split(",")]
    pockets = []

    for line in lines[header_idx + 1 :]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < len(headers):
            continue

        row = dict(zip(headers, parts))
        try:
            rank = int(row.get("rank", 1))
            name = row.get("name", f"pocket_{rank}")
            score = float(row.get("score", 0.0))
            probability = float(row.get("probability", 0.0))
            cx = float(row.get("center_x", 0.0))
            cy = float(row.get("center_y", 0.0))
            cz = float(row.get("center_z", 0.0))

            residue_ids_str = row.get("residue_ids", "")
            raw_ids = [rid.strip() for rid in residue_ids_str.split() if rid.strip()]

            res_1based = []
            res_0based = []
            for rid in raw_ids:
                # Format: "A_745" or "745" or "A:745"
                m = re.search(r"(\d+)", rid)
                if m:
                    val = int(m.group(1))
                    res_1based.append(val)
                    res_0based.append(val - 1)

            pockets.append(
                {
                    "rank": rank,
                    "name": name,
                    "score": score,
                    "probability": probability,
                    "center": (cx, cy, cz),
                    "residue_indices_1based": res_1based,
                    "residue_indices_0based": res_0based,
                    "raw_residue_ids": raw_ids,
                }
            )
        except (ValueError, KeyError) as e:
            logger.debug("Skipping unparseable P2Rank line '%s': %s", line, e)
            continue

    pockets.sort(key=lambda x: x["rank"])
    return pockets


def extract_pocket_residue_mask(
    seq_len: int,
    pocket_indices: List[int],
    zero_indexed: bool = True,
) -> np.ndarray:
    """Generate a boolean mask array of shape [seq_len] with True at pocket residue positions.

    Args:
        seq_len: Total length of the protein sequence.
        pocket_indices: List of pocket residue indices.
        zero_indexed: If False, pocket_indices are treated as 1-based and shifted by -1.

    Returns:
        np.ndarray of shape [seq_len], dtype=bool.
    """
    mask = np.zeros(seq_len, dtype=bool)
    shift = 0 if zero_indexed else -1
    for idx in pocket_indices:
        adj_idx = idx + shift
        if 0 <= adj_idx < seq_len:
            mask[adj_idx] = True
    return mask


def slice_pocket_embeddings(
    embeddings: Any,
    pocket_indices: List[int],
    zero_indexed: bool = True,
) -> Any:
    """Slice only pocket residue token vectors from full sequence embeddings.

    Preserves global PLM/ESM-2 positional contextualization while pruning non-binding residues,
    boosting SNR and speeding up Cross-Attention computation to O(L_pocket^2).

    Args:
        embeddings: Tensor of shape [B, L, D] or [L, D], or numpy ndarray.
        pocket_indices: List of integer indices indicating pocket residue positions.
        zero_indexed: True if indices are 0-based, False if 1-based.

    Returns:
        Sliced embeddings tensor/array containing only the selected pocket residues.
    """
    if not pocket_indices:
        return embeddings

    shift = 0 if zero_indexed else -1
    valid_indices = [idx + shift for idx in pocket_indices if (idx + shift) >= 0]

    # Handle PyTorch Tensor
    if hasattr(embeddings, "ndim") and hasattr(embeddings, "device"):
        import torch

        seq_len = embeddings.shape[-2]
        clamped_idx = [i for i in valid_indices if i < seq_len]
        if not clamped_idx:
            return embeddings
        idx_tensor = torch.tensor(clamped_idx, dtype=torch.long, device=embeddings.device)
        if embeddings.ndim == 3:
            # [B, L, D] -> [B, L_pocket, D]
            return embeddings.index_select(1, idx_tensor)
        elif embeddings.ndim == 2:
            # [L, D] -> [L_pocket, D]
            return embeddings.index_select(0, idx_tensor)
        else:
            return embeddings

    # Handle Numpy array
    arr = np.asarray(embeddings)
    seq_len = arr.shape[-2]
    clamped_idx = [i for i in valid_indices if i < seq_len]
    if not clamped_idx:
        return arr
    if arr.ndim == 3:
        return arr[:, clamped_idx, :]
    elif arr.ndim == 2:
        return arr[clamped_idx, :]
    return arr
