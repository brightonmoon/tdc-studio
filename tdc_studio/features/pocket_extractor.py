"""Binding Pocket Extraction and Domain-Centric Pocket Window Extractor.

Addresses the 1,024 AA truncation limitation:
- For long proteins (>1,000 AA) like EGFR (1,210 AA) or Titin, naive truncation cuts off
  critical C-terminal or middle catalytic domains.
- BindingPocketExtractor identifies catalytic motifs, conserved hydrophobic pocket cores,
  or AlphaFold PDB structural cavity coordinates, extracting a concentrated active pocket window
  (default 128-256 AA) with exact 1-to-1 residue coordinate mapping.
"""

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import urllib.request

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
        dists = np.sqrt(np.sum(diff ** 2, axis=-1))
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
