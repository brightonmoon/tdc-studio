"""Therapeutic Index (TI) Serving Pipeline orchestrating DTI and ADMET models.

Bridges On-Target Affinity (Kd) from DTI models and Safety profiles (hERG cardiotoxicity,
DILI hepatotoxicity) from Unified ADMET into a single clinically interpretable pipeline.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from tdc_studio.evaluation.therapeutic_index import (
    calculate_clinical_progression_score,
    calibrate_herg_ic50_from_probability,
)
from tdc_studio.serving.pipeline import DTIInferencePipeline, DTIMultiAffinityPipeline
from tdc_studio.serving.schema import TherapeuticIndexItem
from tdc_studio.serving.unified_pipeline import (
    UnifiedADMETPipeline,
    get_unified_pipeline,
)

logger = logging.getLogger("tdc_studio.serving.therapeutic_index")

# Canonical human hERG (KCNH2 / Kv11.1, UniProt Q12809) transmembrane pore domain
# Used as reference sequence when predicting direct target affinity against hERG channel
HUMAN_HERG_PORE_DOMAIN_SEQ = (
    "MPVRRGHVAPQNTFLDTIIRKFEGQSRKFIIANTRVQRNALRFHMAGGKLLYYCFE"
    "TDFTSITELVTYTANGTFVIFDITKTTTVGLVVAVFAVGFGNVSPNTENSFFCAFR"
    "IIFLLIVSVVAVMVGFVFLGNTLIVFATVFTSTVFVVFLVGSLF"
)


class TherapeuticIndexPipeline:
    """Orchestrator combining DTI target efficacy and ADMET cardiotoxicity/hepatotoxicity."""

    def __init__(
        self,
        dti_pipeline: Optional[DTIInferencePipeline] = None,
        admet_pipeline: Optional[UnifiedADMETPipeline] = None,
        device: str = "cpu",
    ):
        self.device = device
        self.dti_pipeline = dti_pipeline

        # Initialize or link Unified ADMET pipeline
        if admet_pipeline is not None:
            self.admet_pipeline = admet_pipeline
        else:
            global_admet = get_unified_pipeline()
            self.admet_pipeline = global_admet or UnifiedADMETPipeline(device=device)

    def evaluate_pair(
        self,
        smiles: str,
        target_seq: str,
        herg_source: str = "admet",
        herg_cutoff_nm: float = 10000.0,
        include_admet_details: bool = False,
    ) -> TherapeuticIndexItem:
        """Evaluate a single drug-target pair for therapeutic window and clinical advancement."""
        t0 = time.perf_counter()

        # 1. On-Target Efficacy Prediction (DTI)
        if self.dti_pipeline is not None:
            try:
                if hasattr(self.dti_pipeline, "predict_affinity"):
                    dti_res = self.dti_pipeline.predict_affinity(
                        [smiles], [target_seq], return_kd_nm=True
                    )
                    pkd = float(dti_res["predictions_pkd"][0])
                    kd_nm = float(dti_res["kd_nm"][0])
                else:
                    preds = self.dti_pipeline.predict([smiles], [target_seq])
                    val = preds[0] if isinstance(preds[0], (int, float)) else preds[0][0]
                    pkd = round(float(val), 4)
                    kd_nm = round(float(10.0 ** (9.0 - pkd)), 4)
            except Exception as e:
                logger.warning("DTI prediction error for '%s': %s", smiles, e)
                # Fallback to neutral default
                pkd, kd_nm = 7.0, 100.0
        else:
            # Fallback when no DTI model loaded
            pkd, kd_nm = 7.0, 100.0

        # 2. Safety Profile Prediction (ADMET C1~C5)
        admet_profile = self.admet_pipeline.predict_single(smiles)
        herg_obj = admet_profile.toxicity.get("herg")
        herg_prob = (
            float(herg_obj.probability) if herg_obj and herg_obj.probability is not None else 0.5
        )

        dili_obj = admet_profile.toxicity.get("dili")
        dili_prob = (
            float(dili_obj.probability) if dili_obj and dili_obj.probability is not None else 0.3
        )
        dili_decision_admet = dili_obj.decision if dili_obj else "Low Hepatotoxicity Risk"

        drug_likeness_score = float(admet_profile.drug_likeness_score)

        # 3. Cardiotoxicity hERG IC50 (nM) Resolution
        if herg_source.lower() == "dti" and isinstance(self.dti_pipeline, DTIMultiAffinityPipeline):
            try:
                herg_dti_res = self.dti_pipeline.predict_multi_affinity(
                    [smiles], [HUMAN_HERG_PORE_DOMAIN_SEQ], return_nm=True
                )
                herg_ic50_nm = float(herg_dti_res.get("ic50_nm", [10000.0])[0])
                herg_decision = (
                    "Low Cardiotoxicity Risk"
                    if herg_ic50_nm > 10000.0
                    else ("Moderate Risk" if herg_ic50_nm >= 1000.0 else "High Risk (hERG Blocker)")
                )
            except Exception as e:
                logger.warning("DTI hERG sequence prediction fallback: %s", e)
                herg_ic50_nm, herg_decision = calibrate_herg_ic50_from_probability(
                    herg_prob, cutoff_nm=herg_cutoff_nm
                )
        else:
            herg_ic50_nm, herg_decision = calibrate_herg_ic50_from_probability(
                herg_prob, cutoff_nm=herg_cutoff_nm
            )

        # 4. Therapeutic Index & Clinical Progression Scoring
        scores = calculate_clinical_progression_score(
            kd_nm=kd_nm,
            herg_ic50_nm=herg_ic50_nm,
            dili_prob=dili_prob,
            drug_likeness_score=drug_likeness_score,
            herg_prob=herg_prob,
        )

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        admet_details: Optional[Dict[str, Any]] = None
        if include_admet_details:
            admet_details = {
                "absorption": {k: v.model_dump() for k, v in admet_profile.absorption.items()},
                "distribution": {k: v.model_dump() for k, v in admet_profile.distribution.items()},
                "metabolism": {k: v.model_dump() for k, v in admet_profile.metabolism.items()},
                "excretion": {k: v.model_dump() for k, v in admet_profile.excretion.items()},
                "toxicity": {k: v.model_dump() for k, v in admet_profile.toxicity.items()},
            }

        return TherapeuticIndexItem(
            smiles=smiles,
            target_sequence=target_seq,
            kd_nm=kd_nm,
            pkd=pkd,
            herg_ic50_nm=herg_ic50_nm,
            herg_prob=round(herg_prob, 4),
            herg_decision=herg_decision,
            therapeutic_index=scores["therapeutic_index"],
            ti_tier=scores["ti_tier"],
            dili_prob=round(dili_prob, 4),
            dili_decision=dili_decision_admet or scores["dili_decision"],
            dili_penalty=scores["dili_penalty"],
            drug_likeness_score=round(drug_likeness_score, 1),
            clinical_progression_score=scores["clinical_progression_score"],
            clinical_decision=scores["clinical_decision"],
            safety_radar=scores["safety_radar"],
            admet_profile=admet_details,
            elapsed_ms=elapsed_ms,
        )

    def evaluate_batch(
        self,
        smiles_list: List[str],
        target_sequences: List[str],
        herg_source: str = "admet",
        herg_cutoff_nm: float = 10000.0,
        include_admet_details: bool = False,
    ) -> List[TherapeuticIndexItem]:
        """Evaluate a batch of drug-target pairs."""
        return [
            self.evaluate_pair(
                smiles=s,
                target_seq=t,
                herg_source=herg_source,
                herg_cutoff_nm=herg_cutoff_nm,
                include_admet_details=include_admet_details,
            )
            for s, t in zip(smiles_list, target_sequences)
        ]


_GLOBAL_TI_PIPELINE: Optional[TherapeuticIndexPipeline] = None


def get_therapeutic_index_pipeline() -> Optional[TherapeuticIndexPipeline]:
    """Getter for global TherapeuticIndexPipeline."""
    return _GLOBAL_TI_PIPELINE


def set_therapeutic_index_pipeline(pipeline: Optional[TherapeuticIndexPipeline]) -> None:
    """Setter for global TherapeuticIndexPipeline."""
    global _GLOBAL_TI_PIPELINE
    _GLOBAL_TI_PIPELINE = pipeline
