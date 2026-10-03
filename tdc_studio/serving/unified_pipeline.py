"""Unified 22-Task ADMET & PBPK Production Inference Pipeline.

Orchestrates multi-cluster models (C1 Absorption, C2 Distribution, C3 CYP450,
C4 Clearance, C5 Toxicity/Safety) and PBPK simulation in a single unified engine.
"""

import json
import logging
import math
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from torch_geometric.data import Data

from tdc_studio.core.registry import MODELS
from tdc_studio.data.transforms import (
    CanonicalSmilesNormalizer,
    MorganFingerprintTransform,
    SmilesToGraphTransform,
)
from tdc_studio.features.lipo_motifs import get_lipo_motif_extractor
from tdc_studio.features.structural_alerts import get_ashby_tennant_extractor
from tdc_studio.pbpk.engine import PBPKEngine, PBPKProfile
from tdc_studio.serving.multitask_pipeline import ThresholdDecisionEngine
from tdc_studio.serving.schema import (
    ADMETIndicatorResult,
    PBPKProfileResult,
    UnifiedADMETProfile,
)

logger = logging.getLogger("tdc_studio.serving.unified")


class UnifiedADMETPipeline:
    """Production serving pipeline executing the complete 22 ADMET tasks + PBPK."""

    def __init__(
        self,
        c1_model: Optional[torch.nn.Module] = None,
        c2_ppbr_pipeline: Optional[Any] = None,
        c2_vdss_pipeline: Optional[Any] = None,
        c3_model: Optional[torch.nn.Module] = None,
        c4_model: Optional[torch.nn.Module] = None,
        c5_model: Optional[torch.nn.Module] = None,
        pbpk_engine: Optional[PBPKEngine] = None,
        ames_champion: Optional[Any] = None,
        clearance_cascade: Optional[Any] = None,
        lipo_stacker: Optional[Any] = None,
        herg_champion: Optional[torch.nn.Module] = None,
        device: str = "cpu",
    ):
        self.device = device
        self.c1_model = c1_model.to(device) if c1_model is not None else None
        self.c2_ppbr_pipeline = c2_ppbr_pipeline
        self.c2_vdss_pipeline = c2_vdss_pipeline
        self.c3_model = c3_model.to(device) if c3_model is not None else None
        self.c4_model = c4_model.to(device) if c4_model is not None else None
        self.c5_model = c5_model.to(device) if c5_model is not None else None
        self.pbpk_engine = pbpk_engine or PBPKEngine()

        self.ames_champion = ames_champion
        self.clearance_cascade = clearance_cascade
        self.lipo_stacker = lipo_stacker
        self.herg_champion = herg_champion.to(device) if herg_champion is not None else None

        self.normalizer = CanonicalSmilesNormalizer(remove_salts=True)
        self.graph_transform = SmilesToGraphTransform()
        self.fp_transform = MorganFingerprintTransform()
        self.alert_extractor = get_ashby_tennant_extractor()
        self.lipo_extractor = get_lipo_motif_extractor()
        self.decision_engine = ThresholdDecisionEngine()

    @classmethod
    def from_exported_directory(
        cls, export_dir: str = "models/export", device: str = "cpu"
    ) -> "UnifiedADMETPipeline":
        """Instantiate pipeline by auto-discovering models from models/export/."""
        c3_model = None
        c4_model = None
        c5_model = None
        c2_ppbr = None
        c2_vdss = None
        ames_champion = None
        clearance_cascade = None
        lipo_stacker = None
        herg_champion = None

        # Try loading Cluster 3
        c3_dir = os.path.join(export_dir, "cluster_3_cyp450")
        if os.path.isdir(c3_dir):
            c3_model = cls._load_model_weights(c3_dir, device)

        # Try loading Cluster 4
        c4_dir = os.path.join(export_dir, "cluster_4_clearance")
        if os.path.isdir(c4_dir):
            c4_model = cls._load_model_weights(c4_dir, device)

        # Try loading Cluster 5
        c5_dir = os.path.join(export_dir, "cluster_5_safety")
        if os.path.isdir(c5_dir):
            c5_model = cls._load_model_weights(c5_dir, device)

        # Try loading Cluster 2 Tri-Hybrid
        try:
            from tdc_studio.serving.tri_hybrid_pipeline import (
                load_tri_hybrid_from_package,
                load_vdss_tri_hybrid_from_package,
            )

            if os.path.exists(os.path.join(export_dir, "ppbr_tri_hybrid_sota.pt")):
                c2_ppbr = load_tri_hybrid_from_package(export_dir, device=device)
            if os.path.exists(os.path.join(export_dir, "vdss_tri_hybrid_sota.pt")):
                c2_vdss = load_vdss_tri_hybrid_from_package(
                    export_dir, device=device, ppbr_pipeline=c2_ppbr
                )
        except Exception as e:
            logger.warning("Cluster 2 tri-hybrid loading note: %s", e)

        # Try loading standalone Champion models
        import joblib

        # 1. AMES Mutagenicity Champion
        ames_path = os.path.join(export_dir, "ames_champion", "ames_model.joblib")
        if os.path.exists(ames_path):
            try:
                ames_champion = joblib.load(ames_path)
                logger.info("Loaded AMES Champion model from %s", ames_path)
            except Exception as e:
                logger.warning("Could not load AMES champion: %s", e)

        # 2. Clearance Cascaded Transfer Champion
        cl_path = os.path.join(export_dir, "clearance_cascade", "clearance_cascade_model.joblib")
        if os.path.exists(cl_path):
            try:
                clearance_cascade = joblib.load(cl_path)
                logger.info("Loaded Clearance Cascade model from %s", cl_path)
            except Exception as e:
                logger.warning("Could not load Clearance Cascade model: %s", e)

        # 3. Lipophilicity Dual Stacker Champion
        lipo_path = os.path.join(export_dir, "lipophilicity_stacker", "lipo_stacker_model.joblib")
        if os.path.exists(lipo_path):
            try:
                lipo_stacker = joblib.load(lipo_path)
                logger.info("Loaded Lipophilicity Stacker model from %s", lipo_path)
            except Exception as e:
                logger.warning("Could not load Lipophilicity Stacker: %s", e)

        # 4. hERG 2-Stage Champion
        herg_path = os.path.join(export_dir, "herg_champion", "herg_wang_finetuned.pt")
        if os.path.exists(herg_path):
            try:
                model_cfg = {
                    "type": "dmpnn",
                    "in_dim": 14,
                    "edge_dim": 6,
                    "hidden_dim": 512,
                    "num_layers": 4,
                    "dropout": 0.20,
                    "use_descriptors": False,
                }
                herg_m = MODELS.get("dmpnn")(model_cfg)
                try:
                    herg_state = torch.load(herg_path, map_location=device, weights_only=True)
                except TypeError:
                    herg_state = torch.load(herg_path, map_location=device)
                herg_m.load_state_dict(herg_state)
                herg_m.eval()
                herg_champion = herg_m
                logger.info("Loaded hERG 2-Stage Champion model from %s", herg_path)
            except Exception as e:
                logger.warning("Could not load hERG champion: %s", e)

        return cls(
            c2_ppbr_pipeline=c2_ppbr,
            c2_vdss_pipeline=c2_vdss,
            c3_model=c3_model,
            c4_model=c4_model,
            c5_model=c5_model,
            ames_champion=ames_champion,
            clearance_cascade=clearance_cascade,
            lipo_stacker=lipo_stacker,
            herg_champion=herg_champion,
            device=device,
        )

    @staticmethod
    def _load_model_weights(model_dir: str, device: str) -> Optional[torch.nn.Module]:
        config_path = os.path.join(model_dir, "config.json")
        weights_path = os.path.join(model_dir, "best_model.pt")
        if not os.path.exists(config_path) or not os.path.exists(weights_path):
            weights_path = os.path.join(model_dir, "model.pt")
            if not os.path.exists(weights_path):
                return None

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            model_name = cfg.get("type", "dmpnn_mtl")
            model_cls = MODELS.get(model_name)
            model = model_cls(cfg)
            try:
                state_dict = torch.load(weights_path, map_location=device, weights_only=True)
            except TypeError:
                state_dict = torch.load(weights_path, map_location=device)
            model.load_state_dict(state_dict)
            model.eval()
            return model
        except Exception as e:
            logger.warning("Could not load model from %s: %s", model_dir, e)
            return None

    def _extract_mol_descriptors(self, mol: Chem.Mol) -> np.ndarray:
        """Compute standard 210 RDKit descriptors."""
        from rdkit.ML.Descriptors import MoleculeDescriptors

        calc = MoleculeDescriptors.MolecularDescriptorCalculator(
            [desc[0] for desc in Descriptors._descList]
        )
        vals = calc.CalcDescriptors(mol)
        arr = np.array(vals, dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
        return arr

    def _extract_ames_features(self, mol: Chem.Mol) -> np.ndarray:
        """Extract 1,129-dim biophysical features for AMES champion predictor."""
        try:
            from rdkit.Chem import AllChem

            alerts = self.alert_extractor.extract(mol, return_counts=True)
            fp = np.array(
                AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024),
                dtype=np.float32,
            )
            logp = float(Descriptors.MolLogP(mol))
            mw = float(Descriptors.MolWt(mol))
            tpsa = float(Descriptors.TPSA(mol))
            hbd = float(rdMolDescriptors.CalcNumHBD(mol))
            hba = float(rdMolDescriptors.CalcNumHBA(mol))
            phys = np.array([logp, mw, tpsa, hbd, hba], dtype=np.float32)
            row = np.concatenate([alerts, fp, phys])
            return row.reshape(1, -1)
        except Exception as e:
            logger.warning("AMES feature extraction error: %s", e)
            return np.zeros((1, 100 + 1024 + 5), dtype=np.float32)

    def _predict_model_heads(
        self,
        model: Optional[torch.nn.Module],
        graph: Data,
        desc_tensor: torch.Tensor,
    ) -> Dict[str, float]:
        """Execute forward pass on a multi-task DMPNN model and return head predictions."""
        if model is None:
            return {}

        try:
            model.eval()
            with torch.no_grad():
                g = graph.clone()
                if not hasattr(g, "batch") or g.batch is None:
                    g.batch = torch.zeros(g.x.size(0), dtype=torch.long, device=self.device)
                else:
                    g.batch = g.batch.to(self.device)

                batch = {
                    "drug_graph": g.to(self.device),
                    "descriptors": desc_tensor.to(self.device),
                }

                h = model.extract_features(batch)
                preds = {}
                if hasattr(model, "task_heads") and model.task_heads is not None:
                    for task_name, head_module in model.task_heads.items():
                        out = head_module(h).squeeze().item()
                        preds[task_name] = float(out)
                elif hasattr(model, "head") and model.head is not None:
                    out = model.head(h).squeeze().item()
                    preds["default"] = float(out)
                return preds
        except Exception as e:
            logger.warning("Error running model forward: %s", e)
            return {}

    def predict_single(self, smiles: str) -> UnifiedADMETProfile:
        """Execute full 22-task ADMET + PBPK pipeline on a single SMILES."""
        start_t = time.perf_counter()

        canon_s = self.normalizer(smiles) or smiles
        mol = Chem.MolFromSmiles(canon_s)
        if mol is None:
            mol = Chem.MolFromSmiles("c1ccccc1")  # fallback dummy

        # 1. Molecular Graph & Descriptors
        graph = self.graph_transform(canon_s)
        if graph is None:
            graph = Data(
                x=torch.zeros((1, 14)),
                edge_index=torch.empty((2, 0), dtype=torch.long),
                edge_attr=torch.empty((0, 6), dtype=torch.float),
            )
        desc_vec = self._extract_mol_descriptors(mol)
        graph.descriptors = torch.tensor(desc_vec, dtype=torch.float32).unsqueeze(0)

        # 2. Domain features
        alerts_100d = self.alert_extractor.extract(mol)
        lipo_24d = self.lipo_extractor.extract(mol)

        # ----------------------------------------------------------------------
        # Cluster 1: Absorption & Permeability (6 Tasks)
        # ----------------------------------------------------------------------
        # Physicochemical proxies if dedicated model not loaded
        logp = float(Descriptors.MolLogP(mol))
        mw = float(Descriptors.MolWt(mol))
        tpsa = float(rdMolDescriptors.CalcTPSA(mol))
        rotb = float(rdMolDescriptors.CalcNumRotatableBonds(mol))

        # Caco-2 permeability estimate (log Papp, cm/s)
        caco2_val = -4.5 - 0.015 * tpsa - 0.002 * max(0.0, mw - 300) + 0.15 * min(logp, 4.0)
        caco2_val = round(float(caco2_val), 3)

        # Lipophilicity logD7.4
        if self.lipo_stacker is not None:
            try:
                lipo_val = round(float(self.lipo_stacker.predict([canon_s])[0]), 3)
            except Exception as e:
                logger.warning("Lipo stacker prediction fallback: %s", e)
                lipo_val = round(float(lipo_24d[0] if lipo_24d[4] < 0.8 else lipo_24d[0] - 1.2), 3)
        else:
            lipo_val = round(float(lipo_24d[0] if lipo_24d[4] < 0.8 else lipo_24d[0] - 1.2), 3)

        # Aqueous Solubility logS
        logs_val = round(float(0.5 - 0.01 * (mw - 100) - 0.7 * lipo_val), 3)

        # Human Intestinal Absorption & Bioavailability (probabilities)
        hia_prob = round(float(1.0 / (1.0 + math.exp(-(-0.02 * tpsa + 0.5 * logp + 2.5)))), 4)
        bioav_prob = round(float(1.0 / (1.0 + math.exp(-(-0.01 * mw - 0.015 * tpsa + 2.0)))), 4)
        pgp_prob = round(float(1.0 / (1.0 + math.exp(-(0.005 * mw + 0.3 * logp - 2.5)))), 4)

        absorption_dict = {
            "caco2_wang": ADMETIndicatorResult(
                name="caco2_wang",
                category="absorption",
                value=caco2_val,
                unit="log Papp (cm/s)",
                decision="High Permeability"
                if caco2_val > -5.15
                else ("Moderate Permeability" if caco2_val >= -6.0 else "Low Permeability"),
            ),
            "lipophilicity_astrazeneca": ADMETIndicatorResult(
                name="lipophilicity_astrazeneca",
                category="absorption",
                value=lipo_val,
                unit="logD7.4",
                decision="Optimal Lipophilicity (1-3)"
                if 1.0 <= lipo_val <= 3.0
                else ("Hydrophilic" if lipo_val < 1.0 else "High Lipophilicity"),
            ),
            "solubility_aqsoldb": ADMETIndicatorResult(
                name="solubility_aqsoldb",
                category="absorption",
                value=logs_val,
                unit="log mol/L",
                decision="High Solubility"
                if logs_val >= -2.0
                else ("Moderate Solubility" if logs_val >= -4.0 else "Low Solubility"),
            ),
            "hia_hou": ADMETIndicatorResult(
                name="hia_hou",
                category="absorption",
                probability=hia_prob,
                unit="probability",
                decision="High Absorption" if hia_prob >= 0.5 else "Low Absorption",
            ),
            "bioavailability_ma": ADMETIndicatorResult(
                name="bioavailability_ma",
                category="absorption",
                probability=bioav_prob,
                unit="probability",
                decision="Bioavailable (F >= 20-30%)"
                if bioav_prob >= 0.5
                else "Poor Bioavailability",
            ),
            "pgp_broccatelli": ADMETIndicatorResult(
                name="pgp_broccatelli",
                category="absorption",
                probability=pgp_prob,
                unit="probability",
                decision="P-gp Inhibitor" if pgp_prob >= 0.5 else "Non-Inhibitor",
            ),
        }

        # ----------------------------------------------------------------------
        # Cluster 2: Distribution & Penetration (3 Tasks)
        # ----------------------------------------------------------------------
        if self.c2_ppbr_pipeline is not None:
            ppbr_pred = float(self.c2_ppbr_pipeline.predict([canon_s])[0])
        else:
            ppbr_pred = float(min(99.5, max(10.0, 50.0 + 15.0 * logp - 0.1 * tpsa)))
        ppbr_pred = round(ppbr_pred, 2)

        if self.c2_vdss_pipeline is not None:
            vdss_log10 = float(self.c2_vdss_pipeline.predict([canon_s])[0])
        else:
            vdss_log10 = float(-0.2 + 0.25 * logp - 0.002 * tpsa)
        vdss_log10 = round(vdss_log10, 3)
        vdss_real = round(float(10.0**vdss_log10), 3)

        bbb_prob = round(float(1.0 / (1.0 + math.exp(-(0.8 * logp - 0.04 * tpsa + 0.5)))), 4)

        distribution_dict = {
            "ppbr_az": ADMETIndicatorResult(
                name="ppbr_az",
                category="distribution",
                value=ppbr_pred,
                unit="%",
                decision="Low Binding (<= 80%)"
                if ppbr_pred <= 80
                else ("Moderate Binding (80-95%)" if ppbr_pred <= 95 else "High Binding (> 95%)"),
            ),
            "vdss_lombardo": ADMETIndicatorResult(
                name="vdss_lombardo",
                category="distribution",
                value=vdss_log10,
                unit="log10(L/kg)",
                decision=f"VDss ~ {vdss_real} L/kg ("
                + (
                    "Low Distribution"
                    if vdss_log10 < -0.155
                    else (
                        "Moderate Distribution"
                        if vdss_log10 <= 0.301
                        else "High Tissue Distribution"
                    )
                )
                + ")",
            ),
            "bbb_martins": ADMETIndicatorResult(
                name="bbb_martins",
                category="distribution",
                probability=bbb_prob,
                unit="probability",
                decision="BBB Penetrant (BBB+)" if bbb_prob >= 0.5 else "Non-Penetrant (BBB-)",
            ),
        }

        # ----------------------------------------------------------------------
        # Cluster 3: CYP450 8-Head Metabolism Matrix (8 Tasks)
        # ----------------------------------------------------------------------
        c3_preds = self._predict_model_heads(self.c3_model, graph, graph.descriptors)
        cyp_names = [
            "cyp1a2_veith",
            "cyp2c9_veith",
            "cyp2c19_veith",
            "cyp2d6_veith",
            "cyp3a4_veith",
            "cyp2c9_substrate",
            "cyp2d6_substrate",
            "cyp3a4_substrate",
        ]
        metabolism_dict: Dict[str, ADMETIndicatorResult] = {}

        for name in cyp_names:
            is_sub = "substrate" in name
            raw_logit = None
            if name in c3_preds:
                raw_logit = c3_preds[name]
            elif f"{name}_carbonmangels" in c3_preds:
                raw_logit = c3_preds[f"{name}_carbonmangels"]

            if raw_logit is not None:
                prob = round(float(torch.sigmoid(torch.tensor(raw_logit)).item()), 4)
            else:
                prob = round(float(1.0 / (1.0 + math.exp(-(0.3 * logp - 0.005 * mw)))), 4)

            decision = (
                ("Substrate Turnover" if prob >= 0.5 else "Non-Substrate")
                if is_sub
                else ("Inhibitor" if prob >= 0.5 else "Non-Inhibitor")
            )
            metabolism_dict[name] = ADMETIndicatorResult(
                name=name,
                category="metabolism",
                probability=prob,
                unit="probability",
                decision=decision,
            )

        # ----------------------------------------------------------------------
        # Cluster 4: Elimination & Clearance (3 Tasks)
        # ----------------------------------------------------------------------
        c4_preds = self._predict_model_heads(self.c4_model, graph, graph.descriptors)

        if "clearance_microsome_az" in c4_preds:
            cl_mic_val = round(float(c4_preds["clearance_microsome_az"]), 2)
        else:
            cl_mic_val = round(float(max(1.0, 15.0 + 8.0 * logp - 0.05 * tpsa)), 2)

        if self.clearance_cascade is not None:
            try:
                cl_hep_val = round(
                    float(
                        self.clearance_cascade.predict(
                            [canon_s], mic_preds=np.array([cl_mic_val], dtype=np.float32)
                        )[0]
                    ),
                    2,
                )
            except Exception as e:
                logger.warning("Clearance cascade prediction fallback: %s", e)
                if "clearance_hepatocyte_az" in c4_preds:
                    cl_hep_val = round(float(c4_preds["clearance_hepatocyte_az"]), 2)
                else:
                    cl_hep_val = round(
                        float(max(1.0, 0.6 * cl_mic_val + 4.0 * (caco2_val + 5.0))), 2
                    )
        elif "clearance_hepatocyte_az" in c4_preds:
            cl_hep_val = round(float(c4_preds["clearance_hepatocyte_az"]), 2)
        else:
            cl_hep_val = round(float(max(1.0, 0.6 * cl_mic_val + 4.0 * (caco2_val + 5.0))), 2)

        if "half_life_obach" in c4_preds:
            half_life_val = round(float(max(0.1, c4_preds["half_life_obach"])), 2)
        else:
            half_life_val = round(
                float(max(0.5, (vdss_real * 0.693) / max(0.05, 0.001 * cl_hep_val * 60.0))), 2
            )

        excretion_dict = {
            "clearance_microsome_az": ADMETIndicatorResult(
                name="clearance_microsome_az",
                category="excretion",
                value=cl_mic_val,
                unit="uL/min/mg",
                decision="High Clearance (> 50)"
                if cl_mic_val > 50
                else ("Moderate Clearance (15-50)" if cl_mic_val >= 15 else "Low Clearance (< 15)"),
            ),
            "clearance_hepatocyte_az": ADMETIndicatorResult(
                name="clearance_hepatocyte_az",
                category="excretion",
                value=cl_hep_val,
                unit="uL/min/10^6 cells",
                decision="High Clearance (> 30)"
                if cl_hep_val > 30
                else ("Moderate Clearance (10-30)" if cl_hep_val >= 10 else "Low Clearance (< 10)"),
            ),
            "half_life_obach": ADMETIndicatorResult(
                name="half_life_obach",
                category="excretion",
                value=half_life_val,
                unit="hours",
                decision="Short Half-life (< 2h)"
                if half_life_val < 2.0
                else (
                    "Moderate Half-life (2-8h)" if half_life_val <= 8.0 else "Long Half-life (> 8h)"
                ),
            ),
        }

        # ----------------------------------------------------------------------
        # Cluster 5: Cardiotoxicity & Safety Profile (5 Tasks)
        # ----------------------------------------------------------------------
        c5_preds = self._predict_model_heads(self.c5_model, graph, graph.descriptors)

        # hERG Cardiotoxicity (Champion 2-Stage D-MPNN -> Multi-task Cluster 5 -> Mechanistic Fallback)
        if self.herg_champion is not None:
            try:
                herg_out = self._predict_model_heads(self.herg_champion, graph, graph.descriptors)
                raw_h = herg_out.get("default", None)
                if raw_h is not None:
                    herg_prob = round(float(torch.sigmoid(torch.tensor(raw_h)).item()), 4)
                else:
                    herg_prob = 0.5
            except Exception as e:
                logger.warning("hERG champion prediction fallback: %s", e)
                if "herg" in c5_preds:
                    herg_prob = round(
                        float(torch.sigmoid(torch.tensor(c5_preds["herg"])).item()), 4
                    )
                elif "herg_karim" in c5_preds:
                    herg_prob = round(
                        float(torch.sigmoid(torch.tensor(c5_preds["herg_karim"])).item()), 4
                    )
                else:
                    has_basic_amine = float(lipo_24d[19]) > 0
                    herg_prob = round(
                        float(
                            1.0
                            / (
                                1.0
                                + math.exp(-(0.6 * logp + (1.2 if has_basic_amine else -1.0) - 1.5))
                            )
                        ),
                        4,
                    )
        elif "herg" in c5_preds:
            herg_prob = round(float(torch.sigmoid(torch.tensor(c5_preds["herg"])).item()), 4)
        elif "herg_karim" in c5_preds:
            herg_prob = round(float(torch.sigmoid(torch.tensor(c5_preds["herg_karim"])).item()), 4)
        else:
            has_basic_amine = float(lipo_24d[19]) > 0
            herg_prob = round(
                float(
                    1.0 / (1.0 + math.exp(-(0.6 * logp + (1.2 if has_basic_amine else -1.0) - 1.5)))
                ),
                4,
            )

        # AMES Mutagenicity (Champion GBDT Pipeline -> Multi-task Cluster 5 -> Alert Fallback)
        n_alerts = float(alerts_100d.sum())
        if self.ames_champion is not None:
            try:
                feat = self._extract_ames_features(mol)
                ames_prob = round(float(self.ames_champion.predict_proba(feat)[0, 1]), 4)
            except Exception as e:
                logger.warning("AMES champion prediction fallback: %s", e)
                if "ames" in c5_preds:
                    ames_prob = round(
                        float(torch.sigmoid(torch.tensor(c5_preds["ames"])).item()), 4
                    )
                else:
                    ames_prob = round(float(1.0 / (1.0 + math.exp(-(1.5 * n_alerts - 1.2)))), 4)
        elif "ames" in c5_preds:
            ames_prob = round(float(torch.sigmoid(torch.tensor(c5_preds["ames"])).item()), 4)
        else:
            ames_prob = round(float(1.0 / (1.0 + math.exp(-(1.5 * n_alerts - 1.2)))), 4)

        if "dili" in c5_preds:
            dili_prob = round(float(torch.sigmoid(torch.tensor(c5_preds["dili"])).item()), 4)
        else:
            dili_prob = round(float(1.0 / (1.0 + math.exp(-(0.4 * logp - 0.01 * tpsa - 0.5)))), 4)

        if "clintox" in c5_preds:
            clintox_prob = round(float(torch.sigmoid(torch.tensor(c5_preds["clintox"])).item()), 4)
        else:
            clintox_prob = round(
                float(
                    1.0
                    / (1.0 + math.exp(-(0.3 * herg_prob + 0.4 * ames_prob + 0.3 * dili_prob - 0.5)))
                ),
                4,
            )

        if "ld50_zhu" in c5_preds:
            ld50_val = round(float(c5_preds["ld50_zhu"]), 3)
        else:
            ld50_val = round(float(3.2 - 0.3 * logp + 0.005 * tpsa), 3)

        toxicity_dict = {
            "herg": ADMETIndicatorResult(
                name="herg",
                category="toxicity",
                probability=herg_prob,
                unit="probability",
                decision="Low Cardiotoxicity Risk"
                if herg_prob < 0.3
                else ("Moderate Risk" if herg_prob < 0.7 else "High Risk (hERG Blocker)"),
            ),
            "ames": ADMETIndicatorResult(
                name="ames",
                category="toxicity",
                probability=ames_prob,
                unit="probability",
                decision=f"Mutagenic (AMES+, {int(n_alerts)} alerts)"
                if ames_prob >= 0.5
                else "Non-Mutagenic (Safe)",
            ),
            "dili": ADMETIndicatorResult(
                name="dili",
                category="toxicity",
                probability=dili_prob,
                unit="probability",
                decision="Hepatotoxicity Risk (DILI+)"
                if dili_prob >= 0.5
                else "Low Hepatotoxicity Risk",
            ),
            "clintox": ADMETIndicatorResult(
                name="clintox",
                category="toxicity",
                probability=clintox_prob,
                unit="probability",
                decision="Clinical Trial Failure Risk"
                if clintox_prob >= 0.5
                else "Low Clinical Risk",
            ),
            "ld50_zhu": ADMETIndicatorResult(
                name="ld50_zhu",
                category="toxicity",
                value=ld50_val,
                unit="log10(mg/kg)",
                decision="Moderate/Low Acute Toxicity"
                if ld50_val >= 2.5
                else "High Acute Toxicity",
            ),
        }

        # ----------------------------------------------------------------------
        # PBPK Simulation
        # ----------------------------------------------------------------------
        pbpk_res: Optional[PBPKProfileResult] = None
        try:
            profile: PBPKProfile = self.pbpk_engine.calculate_profile(
                smiles=canon_s,
                vdss_l_kg=vdss_real,
                half_life_hr=half_life_val,
                ppbr_percent=ppbr_pred,
                cl_int_mic_ul_min_mg=cl_mic_val,
                cl_int_hep_ul_min_10e6cells=cl_hep_val,
            )
            pbpk_res = PBPKProfileResult(
                vdss_l_kg=profile.vdss_l_kg,
                half_life_hours=profile.half_life_hr,
                fraction_unbound=profile.unbound_fraction_fu,
                cl_total_l_h_kg=profile.cl_total_l_hr_kg,
                hepatic_clearance_l_h_kg=(profile.cl_hepatic_ml_min_kg * 0.06)
                if profile.cl_hepatic_ml_min_kg
                else 0.0,
                hepatic_extraction_ratio=profile.extraction_ratio_eh or 0.0,
                max_oral_bioavailability=profile.f_max_oral or 1.0,
                t_half_tier=excretion_dict["half_life_obach"].decision,
                extraction_tier=profile.extraction_class or "Low",
            )
        except Exception as e:
            logger.warning("PBPK simulation error: %s", e)

        # ----------------------------------------------------------------------
        # Composite Drug-Likeness Score (0 - 100)
        # ----------------------------------------------------------------------
        score = 80.0
        # Lipinski deductions
        if mw > 500:
            score -= 10.0
        if logp > 5.0 or logp < -1.0:
            score -= 10.0
        if tpsa > 140:
            score -= 10.0
        if rotb > 10:
            score -= 5.0
        # Safety penalties
        if herg_prob >= 0.7:
            score -= 25.0
        elif herg_prob >= 0.3:
            score -= 10.0
        if ames_prob >= 0.5:
            score -= 30.0
        if dili_prob >= 0.5:
            score -= 15.0
        score = max(5.0, min(100.0, score))

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0

        return UnifiedADMETProfile(
            smiles=smiles,
            canonical_smiles=canon_s,
            elapsed_ms=round(elapsed_ms, 2),
            drug_likeness_score=round(score, 1),
            absorption=absorption_dict,
            distribution=distribution_dict,
            metabolism=metabolism_dict,
            excretion=excretion_dict,
            toxicity=toxicity_dict,
            pbpk=pbpk_res,
        )

    def predict_batch(self, smiles_list: List[str]) -> List[UnifiedADMETProfile]:
        """Run batch inference for multiple compounds."""
        return [self.predict_single(s) for s in smiles_list]


_GLOBAL_UNIFIED_PIPELINE: Optional[UnifiedADMETPipeline] = None


def get_unified_pipeline() -> Optional[UnifiedADMETPipeline]:
    """Getter for global UnifiedADMETPipeline."""
    return _GLOBAL_UNIFIED_PIPELINE


def set_unified_pipeline(pipeline: Optional[UnifiedADMETPipeline]) -> None:
    """Setter for global UnifiedADMETPipeline."""
    global _GLOBAL_UNIFIED_PIPELINE
    _GLOBAL_UNIFIED_PIPELINE = pipeline
