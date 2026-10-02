"""Data structures and schemas for Candidate Evaluation Dossier."""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class CandidateSummary:
    """Core physicochemical and structural profile."""

    smiles: str
    canonical_smiles: str
    molecular_formula: str = ""
    molecular_weight: float = 0.0
    logp: float = 0.0
    tpsa: float = 0.0
    hbd: int = 0
    hba: int = 0
    rotatable_bonds: int = 0
    sa_score: float = 0.0
    structure_svg: str = ""  # Inline Base64 or raw SVG


@dataclass
class DossierDataPayload:
    """Comprehensive aggregated data payload for one-click candidate evaluation."""

    candidate: CandidateSummary
    filter_results: Dict[str, Any]  # PAINS, Brenk, Ro5, Veber
    admet_profile: Dict[str, Any]  # 25 ADMET indicators
    therapeutic_index: Optional[Dict[str, Any]] = None  # CDI, hERG safety margin, PBPK Cmax
    pbpk_simulation: Optional[Dict[str, Any]] = None  # Cmax, AUC, t1/2, clearance
    retrosynthesis: Optional[Dict[str, Any]] = None  # Solved, steps, yield, starting materials
    target_attention: Optional[Dict[str, Any]] = None  # Target residue binding hotspots
    created_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire dossier payload to serializable dict."""
        return {
            "candidate": {
                "smiles": self.candidate.smiles,
                "canonical_smiles": self.candidate.canonical_smiles,
                "molecular_weight": round(self.candidate.molecular_weight, 2),
                "logp": round(self.candidate.logp, 2),
                "tpsa": round(self.candidate.tpsa, 2),
                "hbd": self.candidate.hbd,
                "hba": self.candidate.hba,
                "rotatable_bonds": self.candidate.rotatable_bonds,
                "sa_score": round(self.candidate.sa_score, 2),
            },
            "filters": self.filter_results,
            "admet": self.admet_profile,
            "therapeutic_index": self.therapeutic_index,
            "pbpk": self.pbpk_simulation,
            "retrosynthesis": self.retrosynthesis,
            "target_attention": self.target_attention,
            "created_at": self.created_at,
        }
