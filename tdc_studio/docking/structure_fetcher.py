"""Automatic Structure Downloader and Caching Client for AlphaFold DB and RCSB PDB.

Provides:
- fetch_alphafold: Retrieves AlphaFold predicted .pdb models from EMBL-EBI for UniProt IDs.
- fetch_rcsb: Retrieves high-resolution crystallographic .pdb models from RCSB PDB.
- fetch_structure: Auto-detects identifier type (4-letter PDB code vs UniProt ID) and retrieves structure.
- Local disk caching to avoid redundant network calls.
"""

import logging
from pathlib import Path
import re
from typing import Optional, Tuple
import urllib.error
import urllib.request

logger = logging.getLogger("tdc_studio.docking.fetcher")


class StructureFetcher:
    """Client fetching and caching 3D protein structures from AlphaFold DB and RCSB PDB."""

    def __init__(self, cache_dir: Optional[str] = None):
        self.base_cache_dir = Path(cache_dir or "data/structures")
        self.af_dir = self.base_cache_dir / "alphafold"
        self.rcsb_dir = self.base_cache_dir / "rcsb"
        self.af_dir.mkdir(parents=True, exist_ok=True)
        self.rcsb_dir.mkdir(parents=True, exist_ok=True)

    def fetch_alphafold(self, uniprot_id: str) -> Optional[Path]:
        """Fetch AlphaFold v4 predicted structure PDB file for a UniProt accession.

        Args:
            uniprot_id: UniProt ID (e.g. 'P00533' for human EGFR).

        Returns:
            Path to local cached .pdb file, or None if download failed.
        """
        clean_id = uniprot_id.strip().upper()
        local_path = self.af_dir / f"AF-{clean_id}-F1.pdb"
        if local_path.is_file() and local_path.stat().st_size > 100:
            return local_path

        url = f"https://alphafold.ebi.ac.uk/files/AF-{clean_id}-F1-model_v4.pdb"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "TDC-Studio-Docking/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                content = resp.read()
                local_path.write_bytes(content)
                logger.info("Downloaded AlphaFold model for %s -> %s", clean_id, local_path)
                return local_path
        except Exception as exc:
            logger.warning("AlphaFold DB download failed for '%s': %s", clean_id, exc)
            return None

    def fetch_rcsb(self, pdb_id: str) -> Optional[Path]:
        """Fetch experimental X-ray/Cryo-EM crystal structure PDB from RCSB.

        Args:
            pdb_id: 4-character PDB accession (e.g. '1M17', '4HJO').

        Returns:
            Path to local cached .pdb file, or None if download failed.
        """
        clean_id = pdb_id.strip().upper()
        local_path = self.rcsb_dir / f"{clean_id}.pdb"
        if local_path.is_file() and local_path.stat().st_size > 100:
            return local_path

        url = f"https://files.rcsb.org/download/{clean_id}.pdb"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "TDC-Studio-Docking/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                content = resp.read()
                local_path.write_bytes(content)
                logger.info("Downloaded RCSB structure for %s -> %s", clean_id, local_path)
                return local_path
        except Exception as exc:
            logger.warning("RCSB PDB download failed for '%s': %s", clean_id, exc)
            return None

    def fetch_structure(self, identifier: str) -> Tuple[Optional[Path], str]:
        """Automatically identify structure source and retrieve local PDB file.

        Args:
            identifier: Either a 4-char PDB code or a UniProt accession string.

        Returns:
            Tuple of (Path to PDB file or None, source description: 'rcsb', 'alphafold', or 'none').
        """
        clean = identifier.strip().upper()
        # 4-character alphanumeric string without underscores is standard PDB ID (e.g. 1M17, 7KQI)
        if re.match(r"^[0-9][A-Z0-9]{3}$", clean):
            path = self.fetch_rcsb(clean)
            if path is not None:
                return path, "rcsb"

        # Otherwise try UniProt / AlphaFold DB
        path = self.fetch_alphafold(clean)
        if path is not None:
            return path, "alphafold"

        # Secondary fallback: try RCSB in case it was a non-numeric 4-letter ID
        if len(clean) == 4:
            path = self.fetch_rcsb(clean)
            if path is not None:
                return path, "rcsb"

        return None, "none"
