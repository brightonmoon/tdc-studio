"""PBPK Serving Pipeline linking Cluster 2 (PPBR, VDss) and Cluster 4 (Clearance, T1/2)."""

from typing import Any, Dict, List, Optional

from tdc_studio.pbpk.engine import HumanPhysiologicalParams, PBPKEngine


class PBPKServingPipeline:
    """End-to-end PBPK inference pipeline combining distribution and elimination models."""

    def __init__(
        self,
        ppbr_pipeline: Optional[Any] = None,
        vdss_pipeline: Optional[Any] = None,
        clearance_pipeline: Optional[Any] = None,
        phys_params: Optional[HumanPhysiologicalParams] = None,
    ):
        self.ppbr_pipeline = ppbr_pipeline
        self.vdss_pipeline = vdss_pipeline
        self.clearance_pipeline = clearance_pipeline
        self.engine = PBPKEngine(phys_params)

    def predict_pbpk(
        self,
        smiles_list: List[str],
        default_vdss: float = 1.0,
        default_t_half: float = 6.0,
        default_ppbr: float = 85.0,
    ) -> List[Dict[str, Any]]:
        """Predict comprehensive PBPK profile for a batch of SMILES."""
        # 1. Predict PPBR
        ppbr_preds: List[float] = []
        if self.ppbr_pipeline is not None:
            try:
                res = self.ppbr_pipeline.predict(smiles_list)
                ppbr_preds = [float(p) for p in res.get("predictions", [])]
            except Exception:
                ppbr_preds = [default_ppbr] * len(smiles_list)
        if len(ppbr_preds) < len(smiles_list):
            ppbr_preds.extend([default_ppbr] * (len(smiles_list) - len(ppbr_preds)))

        # 2. Predict VDss
        vdss_preds: List[float] = []
        if self.vdss_pipeline is not None:
            try:
                res = self.vdss_pipeline.predict(smiles_list)
                # Note: VDss pipeline outputs log10(L/kg) or linear L/kg
                raw_v = res.get("predictions", [])
                # If values are negative or in log scale, convert 10^v
                vdss_preds = [
                    10.0 ** float(v) if float(v) < 0.5 and float(v) > -3.0 else float(v)
                    for v in raw_v
                ]
            except Exception:
                vdss_preds = [default_vdss] * len(smiles_list)
        if len(vdss_preds) < len(smiles_list):
            vdss_preds.extend([default_vdss] * (len(smiles_list) - len(vdss_preds)))

        # 3. Predict Clearance & Half-life (Cluster 4)
        t_half_preds: List[float] = []
        cl_mic_preds: List[Optional[float]] = []
        cl_hep_preds: List[Optional[float]] = []

        if self.clearance_pipeline is not None:
            try:
                cl_res = self.clearance_pipeline.predict(smiles_list)
                preds_dict = cl_res.get("predictions", {})
                if isinstance(preds_dict, dict):
                    # Multi-task dictionary output
                    t_half_raw = preds_dict.get("half_life_obach", [])
                    t_half_preds = [10.0 ** float(t) for t in t_half_raw]
                    cl_mic_raw = preds_dict.get("clearance_microsome_az", [])
                    cl_mic_preds = [10.0 ** float(c) for c in cl_mic_raw]
                    cl_hep_raw = preds_dict.get("clearance_hepatocyte_az", [])
                    cl_hep_preds = [10.0 ** float(c) for c in cl_hep_raw]
            except Exception:
                pass

        if len(t_half_preds) < len(smiles_list):
            t_half_preds.extend([default_t_half] * (len(smiles_list) - len(t_half_preds)))
        if len(cl_mic_preds) < len(smiles_list):
            cl_mic_preds.extend([None] * (len(smiles_list) - len(cl_mic_preds)))
        if len(cl_hep_preds) < len(smiles_list):
            cl_hep_preds.extend([None] * (len(smiles_list) - len(cl_hep_preds)))

        # 4. Synthesize via PBPKEngine
        results: List[Dict[str, Any]] = []
        for i, smi in enumerate(smiles_list):
            prof = self.engine.calculate_profile(
                smiles=smi,
                vdss_l_kg=vdss_preds[i],
                half_life_hr=t_half_preds[i],
                ppbr_percent=ppbr_preds[i],
                cl_int_mic_ul_min_mg=cl_mic_preds[i],
                cl_int_hep_ul_min_10e6cells=cl_hep_preds[i],
            )
            results.append(prof.to_dict())

        return results
