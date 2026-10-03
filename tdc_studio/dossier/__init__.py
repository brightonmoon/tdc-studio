"""Candidate Evaluation Dossier Generation Module."""

from tdc_studio.dossier.collector import DossierCollector
from tdc_studio.dossier.models import CandidateSummary, DossierDataPayload
from tdc_studio.dossier.renderer import DossierRenderer

__all__ = [
    "DossierCollector",
    "DossierRenderer",
    "DossierDataPayload",
    "CandidateSummary",
]
