"""High-Performance Production Inference Pipeline for PPBR Tri-Hybrid SOTA Stacker.

Integrates:
1. Branch 1: Deep Cross-Task Message Passing GNN (D-MPNN)
2. Branch 2: Biophysical & 3D Conformer Fused GBDT (HistGradientBoosting)
3. Branch 3: Foundation Model Manifold Ridge (ChemBERTa-77M-MTR)
4. Parametric Sigmoid Calibration (α, β) and Optimal Convex Blending
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from torch_geometric.data import Data

from tdc_studio.data.collate import molecule_collate_fn
from tdc_studio.data.transforms import SmilesToGraphTransform
from tdc_studio.models.graph.dmpnn import DMPNNModel
from tdc_studio.models.hybrid.gbdt_blend import (
    compute_biophysical_motifs,
    extract_chemberta_embeddings_batch,
)
from tdc_studio.serving.multitask_pipeline import ThresholdDecisionEngine

_QUAT_N_SMARTS = Chem.MolFromSmarts("[NX4+;!$([NX4+]([O-])=O)]")


class TriHybridInferencePipeline:
    """Production serving pipeline for SOTA Tri-Hybrid Stacker (PPBR)."""

    def __init__(
        self,
        dmpnn_model: torch.nn.Module,
        gbdt_model: Any,
        ridge_model: Any,
        scaler: Any,
        blending_weights: Tuple[float, float, float],
        calibration_params: Tuple[float, float],
        dmpnn_stats: Tuple[float, float] = (3.238, 1.488),
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.dmpnn_model = dmpnn_model.to(self.device)
        self.dmpnn_model.eval()

        self.gbdt_model = gbdt_model
        self.ridge_model = ridge_model
        self.scaler = scaler

        self.w_gnn, self.w_gbdt, self.w_ridge = blending_weights
        self.alpha, self.beta = calibration_params
        self.mu_trn, self.std_trn = dmpnn_stats

        self.graph_transform = SmilesToGraphTransform()
        self.decision_engine = ThresholdDecisionEngine()

    @staticmethod
    def _compute_3d_and_biophysical_motifs(mol: Optional[Chem.Mol]) -> List[float]:
        """Compute 18 biophysical motifs + 6 3D conformer steric parameters."""
        base_14 = compute_biophysical_motifs(mol)
        if mol is None:
            return base_14 + [0.0] * 10

        n_quat_n = float(len(mol.GetSubstructMatches(_QUAT_N_SMARTS)))
        mw = Descriptors.MolWt(mol)
        tpsa = Descriptors.TPSA(mol)
        tpsa_mw_ratio = float(tpsa / max(1.0, mw))
        fsp3 = float(Descriptors.FractionCSP3(mol))
        hac = Descriptors.HeavyAtomCount(mol)
        rot_dens = float(Descriptors.NumRotatableBonds(mol) / max(1.0, hac))

        # 3D Conformer parameters
        pbf, spherocity, asphericity, eccentricity, inert_factor, rog = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        try:
            mol_3d = Chem.AddHs(mol)
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            params.maxAttempts = 5
            cid = AllChem.EmbedMolecule(mol_3d, params)
            if cid >= 0:
                pbf = float(rdMolDescriptors.CalcPBF(mol_3d, confId=cid))
                spherocity = float(rdMolDescriptors.CalcSpherocityIndex(mol_3d, confId=cid))
                asphericity = float(rdMolDescriptors.CalcAsphericity(mol_3d, confId=cid))
                eccentricity = float(rdMolDescriptors.CalcEccentricity(mol_3d, confId=cid))
                inert_factor = float(rdMolDescriptors.CalcInertialShapeFactor(mol_3d, confId=cid))
                rog = float(rdMolDescriptors.CalcRadiusOfGyration(mol_3d, confId=cid))
        except Exception:
            pass

        return base_14 + [
            n_quat_n,
            tpsa_mw_ratio,
            fsp3,
            rot_dens,
            pbf,
            spherocity,
            asphericity,
            eccentricity,
            inert_factor,
            rog,
        ]

    def _extract_tabular_features(
        self, smiles_list: List[str], n_bits: int = 1024
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract 1024 Morgan FP + 210 RDKit Descriptors + 24 Bio/3D + ChemBERTa."""
        features = []
        dummy_mol = Chem.MolFromSmiles("C")
        n_desc = len(Descriptors.CalcMolDescriptors(dummy_mol)) if dummy_mol is not None else 210

        for s in smiles_list:
            mol = Chem.MolFromSmiles(s) if (s and isinstance(s, str)) else None
            if mol is None:
                features.append([0.0] * (n_bits + n_desc + 24))
            else:
                fp = list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits))
                desc_dict = Descriptors.CalcMolDescriptors(mol)
                desc_vals = [
                    0.0
                    if (v is None or np.isnan(v) or np.isinf(v))
                    else float(np.clip(v, -100.0, 100.0))
                    for v in desc_dict.values()
                ]
                bio_3d = self._compute_3d_and_biophysical_motifs(mol)
                features.append(fp + desc_vals + bio_3d)

        X_base = np.array(features, dtype=np.float32)

        # ChemBERTa Embeddings
        chemberta_embs = extract_chemberta_embeddings_batch(smiles_list)
        if chemberta_embs is None or len(chemberta_embs) != len(smiles_list):
            chemberta_embs = np.zeros((len(smiles_list), 384), dtype=np.float32)

        X_fused = np.hstack([X_base, chemberta_embs])
        return X_fused, chemberta_embs

    def _predict_dmpnn(self, smiles_list: List[str]) -> np.ndarray:
        """Run Branch 1 DMPNN Graph inference."""
        batch_items = []
        for sm in smiles_list:
            g = self.graph_transform(sm)
            if g is None:
                g = Data(
                    x=torch.zeros((1, 14)),
                    edge_index=torch.empty((2, 0), dtype=torch.long),
                    edge_attr=torch.empty((0, 6), dtype=torch.float),
                )
            mol = Chem.MolFromSmiles(sm) if (sm and isinstance(sm, str)) else None
            if mol is not None:
                desc_dict = Descriptors.CalcMolDescriptors(mol)
                desc_vals = [
                    0.0
                    if (v is None or np.isnan(v) or np.isinf(v))
                    else float(np.clip(v, -100.0, 100.0))
                    for v in desc_dict.values()
                ]
            else:
                desc_vals = [0.0] * 210

            batch_items.append(
                {
                    "drug_graph": g,
                    "descriptors": torch.tensor(desc_vals, dtype=torch.float32),
                    "drug_smiles_str": sm,
                }
            )

        collated = molecule_collate_fn(batch_items)
        for k, v in collated.items():
            if hasattr(v, "to"):
                collated[k] = v.to(self.device)

        with torch.no_grad():
            preds = self.dmpnn_model(collated)[:, 0].cpu().numpy()

        # De-standardize to thermodynamic logit space
        z_gnn = preds * self.std_trn + self.mu_trn
        return z_gnn

    def predict(
        self, smiles_list: List[str], target_sequences: Optional[List[str]] = None
    ) -> List[float]:
        """Predict PPBR binding percentage (%) for list of SMILES."""
        if not smiles_list:
            return []

        # 1. Branch 1: DMPNN Graph Logits
        z_gnn = self._predict_dmpnn(smiles_list)

        # 2. Branch 2 & 3: Tabular Features & ChemBERTa
        X_fused, emb_bert = self._extract_tabular_features(smiles_list)
        z_gbdt = self.gbdt_model.predict(X_fused)

        # Branch 3: Scaled ChemBERTa Ridge
        emb_sc = self.scaler.transform(emb_bert) if self.scaler is not None else emb_bert
        z_ridge = self.ridge_model.predict(emb_sc)

        # 3. Blending in logit space
        z_blend = self.w_gnn * z_gnn + self.w_gbdt * z_gbdt + self.w_ridge * z_ridge

        # 4. Parametric Sigmoid Calibration: [0, 100]%
        pred_pct = 100.0 / (1.0 + np.exp(-np.clip(self.alpha * z_blend + self.beta, -40.0, 40.0)))
        pred_pct = np.clip(pred_pct, 0.0, 100.0)

        return [round(float(v), 2) for v in pred_pct]

    def predict_detailed(self, smiles_list: List[str]) -> List[Dict[str, Any]]:
        """Return detailed predictions including qualitative medicinal chemistry tiers and branch scores."""
        scores = self.predict(smiles_list)
        results = []
        for s, score in zip(smiles_list, scores):
            eval_res = self.decision_engine.evaluate("ppbr_az", score, "regression")
            results.append(
                {
                    "smiles": s,
                    "ppbr_percent": score,
                    "unit": "%",
                    "decision": eval_res.get("decision", ""),
                    "status": "success",
                }
            )
        return results


def load_tri_hybrid_from_package(
    package_dir: str, device: str = "cpu"
) -> TriHybridInferencePipeline:
    """Load SOTA Tri-Hybrid pipeline from exported directory."""
    import pickle

    weights_path = os.path.join(package_dir, "ppbr_tri_hybrid_sota.pt")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(package_dir, "model.pt")

    package = torch.load(weights_path, map_location=device, weights_only=False)

    dmpnn_cfg = package["dmpnn_config"]
    dmpnn_model = DMPNNModel(dmpnn_cfg)
    dmpnn_model.load_state_dict(package["dmpnn_state_dict"])

    gbdt_model = pickle.loads(package["gbdt_pickle"])
    ridge_model = pickle.loads(package["ridge_pickle"])
    scaler = pickle.loads(package["scaler_pickle"])

    weights = package.get("blending_weights", (0.15, 0.70, 0.15))
    calibration = package.get("calibration_params", (0.85, 0.02))
    stats = package.get("dmpnn_stats", (3.238, 1.488))

    return TriHybridInferencePipeline(
        dmpnn_model=dmpnn_model,
        gbdt_model=gbdt_model,
        ridge_model=ridge_model,
        scaler=scaler,
        blending_weights=weights,
        calibration_params=calibration,
        dmpnn_stats=stats,
        device=device,
    )


def compute_extended_pk_motifs(mol: Optional[Chem.Mol], ppbr_val: float = 90.0) -> List[float]:
    """Compute 24 biophysical + 3D steric motifs + Oie-Tozer fraction unbound features."""
    if mol is None:
        return [0.0] * 30

    base_14 = compute_biophysical_motifs(mol)
    n_quat_n = float(len(mol.GetSubstructMatches(_QUAT_N_SMARTS)))
    mw = Descriptors.MolWt(mol)
    tpsa = Descriptors.TPSA(mol)
    tpsa_mw = float(tpsa / max(1.0, mw))
    fsp3 = float(Descriptors.FractionCSP3(mol))
    hac = Descriptors.HeavyAtomCount(mol)
    rot_dens = float(Descriptors.NumRotatableBonds(mol) / max(1.0, hac))

    # 3D Conformer Steric
    pbf, spherocity, asphericity, eccentricity, inert_factor, rog = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    try:
        mol_3d = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        params.maxAttempts = 5
        cid = AllChem.EmbedMolecule(mol_3d, params)
        if cid >= 0:
            AllChem.MMFFOptimizeMolecule(mol_3d, confId=cid, maxIters=30)
            pbf = float(rdMolDescriptors.CalcPBF(mol_3d, confId=cid))
            spherocity = float(rdMolDescriptors.CalcSpherocityIndex(mol_3d, confId=cid))
            asphericity = float(rdMolDescriptors.CalcAsphericity(mol_3d, confId=cid))
            eccentricity = float(rdMolDescriptors.CalcEccentricity(mol_3d, confId=cid))
            inert_factor = float(rdMolDescriptors.CalcInertialShapeFactor(mol_3d, confId=cid))
            rog = float(rdMolDescriptors.CalcRadiusOfGyration(mol_3d, confId=cid))
    except Exception:
        pass

    # Oie-Tozer Mechanistic PK Features (from PPBR prediction)
    fb = float(np.clip(ppbr_val / 100.0, 1e-4, 1.0 - 1e-4))
    fu = float(1.0 - fb)
    log_fu = float(np.log10(max(1e-4, fu)))
    logit_ppbr = float(np.log(fb / (1.0 - fb)))
    fu_ratio = float(fu / max(1e-4, fb))
    log_fu_ratio = float(np.log10(max(1e-4, fu_ratio)))
    sqrt_fu = float(np.sqrt(fu))

    return base_14 + [
        n_quat_n,
        tpsa_mw,
        fsp3,
        rot_dens,
        pbf,
        spherocity,
        asphericity,
        eccentricity,
        inert_factor,
        rog,
        fu,
        log_fu,
        logit_ppbr,
        fu_ratio,
        log_fu_ratio,
        sqrt_fu,
    ]


class VDssTriHybridInferencePipeline:
    """Production serving pipeline for SOTA Tri-Hybrid Stacker (VDss Lombardo).

    Integrates:
    1. Mechanistic Oie-Tozer unbound fraction (fu) coupled with PPBR SOTA Stacker
    2. Branch 1: Deep ChemBERTa-77M-MTR Foundation Ridge
    3. Branch 2: Biophysical, 3D Conformer & PK-Motif Fused GBDT (HistGradientBoosting)
    4. Branch 3: Deep Cross-Task Message Passing GNN (D-MPNN)
    5. Pure Convex Optimal Blending
    """

    def __init__(
        self,
        gbdt_model: Any,
        ridge_model: Any,
        scaler: Any,
        blending_weights: Tuple[float, float, float] = (0.543, 0.084, 0.374),
        calibration_params: Tuple[float, float] = (1.0, 0.0),
        dmpnn_model: Optional[torch.nn.Module] = None,
        ppbr_pipeline: Optional[TriHybridInferencePipeline] = None,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.gbdt_model = gbdt_model
        self.ridge_model = ridge_model
        self.scaler = scaler
        self.w_gbdt, self.w_ridge, self.w_dmpnn = blending_weights
        self.alpha, self.beta = calibration_params
        self.dmpnn_model = dmpnn_model.to(self.device) if dmpnn_model is not None else None
        if self.dmpnn_model is not None:
            self.dmpnn_model.eval()
        self.ppbr_pipeline = ppbr_pipeline
        self.graph_transform = SmilesToGraphTransform()
        self.decision_engine = ThresholdDecisionEngine()

    def _predict_dmpnn(self, smiles_list: List[str]) -> np.ndarray:
        if self.dmpnn_model is None:
            return np.zeros(len(smiles_list), dtype=np.float32)

        batch_items = []
        for sm in smiles_list:
            g = self.graph_transform(sm)
            if g is None:
                g = Data(
                    x=torch.zeros((1, 14)),
                    edge_index=torch.empty((2, 0), dtype=torch.long),
                    edge_attr=torch.empty((0, 6), dtype=torch.float),
                )
            mol = Chem.MolFromSmiles(sm) if (sm and isinstance(sm, str)) else None
            if mol is not None:
                desc_dict = Descriptors.CalcMolDescriptors(mol)
                desc_vals = [
                    0.0
                    if (v is None or np.isnan(v) or np.isinf(v))
                    else float(np.clip(v, -100.0, 100.0))
                    for v in desc_dict.values()
                ]
            else:
                desc_vals = [0.0] * 210
            batch_items.append(
                {
                    "drug_graph": g,
                    "descriptors": torch.tensor(desc_vals, dtype=torch.float32),
                    "drug_smiles_str": sm,
                }
            )
        collated = molecule_collate_fn(batch_items)
        for k, v in collated.items():
            if hasattr(v, "to"):
                collated[k] = v.to(self.device)
        with torch.no_grad():
            out = self.dmpnn_model(collated)
            v_idx = 2 if out.shape[-1] > 2 else 0
            preds = out[:, v_idx].cpu().numpy()
        return preds

    def _extract_tabular_features(
        self, smiles_list: List[str], ppbr_preds: List[float], n_bits: int = 1024
    ) -> Tuple[np.ndarray, np.ndarray]:
        features = []
        dummy_mol = Chem.MolFromSmiles("C")
        n_desc = len(Descriptors.CalcMolDescriptors(dummy_mol)) if dummy_mol is not None else 210

        for s, ppbr_v in zip(smiles_list, ppbr_preds):
            mol = Chem.MolFromSmiles(s) if (s and isinstance(s, str)) else None
            if mol is None:
                features.append([0.0] * (n_bits + n_desc + 30))
            else:
                fp = list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits))
                desc_dict = Descriptors.CalcMolDescriptors(mol)
                desc_vals = [
                    0.0
                    if (v is None or np.isnan(v) or np.isinf(v))
                    else float(np.clip(v, -100.0, 100.0))
                    for v in desc_dict.values()
                ]
                pk_motifs = compute_extended_pk_motifs(mol, ppbr_v)
                features.append(fp + desc_vals + pk_motifs)

        X_base = np.array(features, dtype=np.float32)
        chemberta_embs = extract_chemberta_embeddings_batch(smiles_list)
        if chemberta_embs is None or len(chemberta_embs) != len(smiles_list):
            chemberta_embs = np.zeros((len(smiles_list), 384), dtype=np.float32)

        X_fused = np.hstack([X_base, chemberta_embs])
        return X_fused, chemberta_embs

    def predict(
        self, smiles_list: List[str], target_sequences: Optional[List[str]] = None
    ) -> List[float]:
        """Predict Volume of Distribution (log10 L/kg) for list of SMILES."""
        if not smiles_list:
            return []

        if self.ppbr_pipeline is not None:
            ppbr_preds = self.ppbr_pipeline.predict(smiles_list)
        else:
            ppbr_preds = [90.0] * len(smiles_list)

        X_fused, emb_bert = self._extract_tabular_features(smiles_list, ppbr_preds)
        p_gbdt = self.gbdt_model.predict(X_fused)

        emb_sc = self.scaler.transform(emb_bert) if self.scaler is not None else emb_bert
        p_ridge = self.ridge_model.predict(emb_sc)

        p_dmpnn = self._predict_dmpnn(smiles_list) if self.w_dmpnn > 0.0 else np.zeros_like(p_gbdt)

        p_blend = (
            self.alpha * (self.w_gbdt * p_gbdt + self.w_ridge * p_ridge + self.w_dmpnn * p_dmpnn)
            + self.beta
        )

        return [round(float(v), 4) for v in p_blend]

    def predict_detailed(self, smiles_list: List[str]) -> List[Dict[str, Any]]:
        """Return detailed predictions including qualitative pharmacology tiers and VDss in L/kg."""
        scores = self.predict(smiles_list)
        results = []
        for s, score in zip(smiles_list, scores):
            eval_res = self.decision_engine.evaluate("vdss_lombardo", score, "regression")
            results.append(
                {
                    "smiles": s,
                    "vdss_log10": score,
                    "vdss_L_kg": eval_res.get("real_vdss_L_kg", round(10.0**score, 3)),
                    "unit": "log10(L/kg)",
                    "decision": eval_res.get("decision", ""),
                    "status": "success",
                }
            )
        return results


def load_vdss_tri_hybrid_from_package(
    package_dir: str,
    device: str = "cpu",
    ppbr_pipeline: Optional[TriHybridInferencePipeline] = None,
) -> VDssTriHybridInferencePipeline:
    """Load SOTA VDss Tri-Hybrid pipeline from exported directory."""
    import pickle

    weights_path = os.path.join(package_dir, "vdss_tri_hybrid_sota.pt")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(package_dir, "model.pt")

    package = torch.load(weights_path, map_location=device, weights_only=False)

    dmpnn_model = None
    if package.get("dmpnn_config") is not None and package.get("dmpnn_state_dict") is not None:
        from tdc_studio.core.registry import MODELS

        dmpnn_cfg = package["dmpnn_config"]
        model_cls = MODELS.get(dmpnn_cfg.get("type", "dmpnn_mtl"))
        dmpnn_model = model_cls(dmpnn_cfg)
        dmpnn_model.load_state_dict(package["dmpnn_state_dict"])
        dmpnn_model.eval()

    gbdt_model = pickle.loads(package["gbdt_pickle"])
    ridge_model = pickle.loads(package["ridge_pickle"])
    scaler = pickle.loads(package["scaler_pickle"])

    weights = package.get("optimal_weights", (0.543, 0.084, 0.374))
    calibration = package.get("calibration", (1.0, 0.0))

    if ppbr_pipeline is None and os.path.exists(
        os.path.join(package_dir, "ppbr_tri_hybrid_sota.pt")
    ):
        try:
            ppbr_pipeline = load_tri_hybrid_from_package(package_dir, device=device)
        except Exception:
            ppbr_pipeline = None

    return VDssTriHybridInferencePipeline(
        gbdt_model=gbdt_model,
        ridge_model=ridge_model,
        scaler=scaler,
        blending_weights=weights,
        calibration_params=calibration,
        dmpnn_model=dmpnn_model,
        ppbr_pipeline=ppbr_pipeline,
        device=device,
    )
