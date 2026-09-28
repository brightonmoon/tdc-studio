"""End-to-End inference pipeline combining molecular featurization and model prediction."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

from tdc_studio.data.collate import molecule_collate_fn
from tdc_studio.data.transforms import (
    MorganFingerprintTransform,
    SequenceTokenizer,
    SmilesToGraphTransform,
    SmilesTokenizer,
)
from tdc_studio.serving.xai_utils import (
    compute_affinity_consistency_score,
    extract_top_contact_atoms,
    extract_top_contact_residues,
    generate_pymol_command,
)


class InferencePipeline:
    """Unified inference engine encapsulating CPU featurization and PyTorch model execution."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cpu",
        is_dta: bool = False,
        modality: str = "graph",
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.is_dta = is_dta
        self.modality = modality.lower()

        self.graph_transform = SmilesToGraphTransform()
        self.smiles_tokenizer = SmilesTokenizer()
        self.fingerprint_transform = MorganFingerprintTransform()
        self.target_tokenizer = SequenceTokenizer() if is_dta else None

    def predict(
        self, smiles_list: List[str], target_seqs: Optional[List[str]] = None
    ) -> List[float]:
        """Run batch inference on raw SMILES and optional targets across modalities."""
        batch_items = []
        for i, sm in enumerate(smiles_list):
            item: dict[str, Any] = {}

            if self.modality == "sequence":
                item["smiles_seq"] = self.smiles_tokenizer(sm)
            elif self.modality == "fingerprint":
                fp = self.fingerprint_transform(sm)
                if fp is None:
                    fp = torch.zeros(2048, dtype=torch.float32)
                item["fingerprint"] = fp
            else:
                # Default to graph
                g = self.graph_transform(sm)
                if g is None:
                    g = Data(
                        x=torch.zeros((1, 14)),
                        edge_index=torch.empty((2, 0), dtype=torch.long),
                        edge_attr=torch.empty((0, 6), dtype=torch.float),
                    )
                item["drug_graph"] = g

            if self.is_dta and target_seqs is not None and i < len(target_seqs):
                item["target_seq"] = self.target_tokenizer(target_seqs[i])

            batch_items.append(item)

        # Collate items
        collated = molecule_collate_fn(batch_items)

        # Move to device
        for k, v in collated.items():
            if hasattr(v, "to"):
                collated[k] = v.to(self.device)

        with torch.no_grad():
            preds = self.model(collated)
            preds_flat = preds.squeeze(-1).detach().cpu().numpy().tolist()

        if isinstance(preds_flat, float):
            return [preds_flat]
        return preds_flat


class DTIInferencePipeline(InferencePipeline):
    """Inference pipeline specialised for Drug-Target Interaction models.

    Extends InferencePipeline to correctly handle dual-input (SMILES + AA seq)
    batches required by GraphDTAModel and future pretrained-encoder variants.

    Supports:
    - Dual featurization: SMILES (graph + raw string) and AA sequence (tokens + raw string)
    - Inverse standardization using fitted dataset statistics (y_mean, y_std, log1p)
    - Direct conversion to pKd (-log10 Kd) and Kd in nanomolar (nM)

    Args:
        model        : Trained GraphDTAModel (or compatible DTA model).
        device       : "cpu" or "cuda".
        aa_max_length: Max AA sequence length for AminoAcidTokenizer (default 1000).
        scaler_meta  : Optional dict containing normalization metadata:
                       {"y_mean": float, "y_std": float, "log_transform": bool}.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cpu",
        aa_max_length: int = 1000,
        scaler_meta: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(model=model, device=device, is_dta=True, modality="graph")
        from tdc_studio.data.transforms import AminoAcidTokenizer

        self.aa_tokenizer = AminoAcidTokenizer(max_length=aa_max_length)
        self.scaler_meta = scaler_meta or {}

        # Pre-create reusable dummy graph to avoid allocations when drug is SMILES-based
        self._dummy_graph = Data(
            x=torch.zeros((1, 14)),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_attr=torch.empty((0, 6), dtype=torch.float),
        )

        # Check if drug encoder is SMILES/sequence based (ChemBERTa) to skip expensive RDKit graph conversion
        self.is_graph_drug = True
        if hasattr(model, "drug_encoder"):
            enc_name = type(model.drug_encoder).__name__.lower()
            if "chembert" in enc_name or "language" in enc_name:
                self.is_graph_drug = False

    def inverse_transform(self, y_norm: float) -> Tuple[float, float]:
        """Convert normalized model prediction to pKd and Kd (nM).

        Returns:
            Tuple of (pKd, Kd_in_nM).
        """
        if self.scaler_meta and "y_mean" in self.scaler_meta:
            y_mean = float(self.scaler_meta.get("y_mean", 0.0))
            y_std = float(self.scaler_meta.get("y_std", 1.0))
            log_transform = bool(self.scaler_meta.get("log_transform", True))

            # Reverse z-score
            y_log = y_norm * y_std + y_mean
            # Reverse log1p (Y was Kd in nM)
            kd_nm = float(np.expm1(y_log)) if log_transform else float(y_log)
            kd_nm = max(kd_nm, 1e-4)

            # pKd = -log10(Kd_in_molar) = 9.0 - log10(Kd_in_nM)
            pkd = float(9.0 - np.log10(kd_nm))
        else:
            # Fallback: assume model outputs pKd or raw normalized score directly
            pkd = float(y_norm)
            bounded_pkd = min(max(pkd, 0.0), 14.0)
            kd_nm = float(10.0 ** (9.0 - bounded_pkd))

        return round(pkd, 4), round(kd_nm, 4)

    def predict(
        self,
        smiles_list: List[str],
        target_seqs: Optional[List[str]] = None,
        return_attention: bool = False,
    ) -> Any:
        """Run DTI affinity prediction for paired (SMILES, AA sequence) inputs.

        Args:
            smiles_list     : List of drug SMILES strings.
            target_seqs     : List of protein AA sequences (same length as smiles_list).
            return_attention: Whether to return cross-attention weights if supported.

        Returns:
            List of raw predictions, or (preds, attn_list) if return_attention=True.
        """
        if target_seqs is None or len(target_seqs) != len(smiles_list):
            raise ValueError(
                "DTIInferencePipeline requires target_seqs of the same length as smiles_list."
            )

        batch_items = []
        for sm, seq in zip(smiles_list, target_seqs):
            clean_smiles = sm.strip()
            clean_seq = seq.strip().upper()

            # Drug graph: skip RDKit parsing if model is string/ChemBERTa-based
            if self.is_graph_drug:
                g = self.graph_transform(clean_smiles)
                if g is None:
                    g = self._dummy_graph
            else:
                g = self._dummy_graph

            # Target sequence tensor
            target_tensor = self.aa_tokenizer(clean_seq)

            batch_items.append({
                "drug_graph":      g,
                "drug_smiles_str": clean_smiles,   # for HuggingFace / ChemBERTa encoder
                "target_seq":      target_tensor,  # for ProteinCNN encoder
                "target_seq_str":  clean_seq,      # for ESM-2 encoder (avoids decode fallback)
            })

        collated = molecule_collate_fn(batch_items)
        for k, v in collated.items():
            if hasattr(v, "to"):
                collated[k] = v.to(self.device)

        raw_attn = None
        with torch.no_grad():
            if return_attention:
                try:
                    model_out = self.model(collated, return_attention=True, return_sequence=True)
                except TypeError:
                    try:
                        model_out = self.model(collated, return_attention=True)
                    except TypeError:
                        model_out = self.model(collated)
            else:
                model_out = self.model(collated)

            if isinstance(model_out, tuple):
                preds, raw_attn = model_out
            else:
                preds = model_out

            preds_arr = preds.squeeze(-1).detach().cpu().numpy()
            if preds_arr.ndim == 0:
                preds_flat = [float(preds_arr)]
            else:
                preds_flat = preds_arr.tolist()

        if return_attention:
            attn_list = []
            batch_size = len(smiles_list)
            if isinstance(raw_attn, dict):
                for b in range(batch_size):
                    item_attn = {}
                    for k, t in raw_attn.items():
                        if isinstance(t, torch.Tensor):
                            t_slice = t[b].detach().cpu().numpy()
                            item_attn[k] = t_slice
                    attn_list.append(item_attn)
            else:
                attn_list = [{} for _ in range(batch_size)]
            return preds_flat, attn_list

        return preds_flat

    def predict_affinity(
        self,
        smiles_list: List[str],
        target_seqs: List[str],
        return_kd_nm: bool = True,
        return_attention: bool = False,
        return_contact_map: bool = False,
        top_k_residues: int = 10,
        return_full_matrix: bool = False,
    ) -> Dict[str, Any]:
        """Run DTI prediction and return structured pKd, Kd (nM), and optional XAI contact map.

        Args:
            smiles_list       : List of drug SMILES strings.
            target_seqs       : List of protein AA sequences.
            return_kd_nm      : Whether to compute Kd in nM.
            return_attention  : Whether to include attention weight maps in output.
            return_contact_map: Whether to extract Top-K residues, atoms, and PyMOL commands.
            top_k_residues    : Number of top contact residues to extract for PyMOL.
            return_full_matrix: Whether to include full 2D float contact map array.

        Returns:
            Dict containing predictions, Kd values, and optional XAI attributes.
        """
        need_attn = return_attention or return_contact_map
        if need_attn:
            raw_preds, attn_list = self.predict(
                smiles_list=smiles_list,
                target_seqs=target_seqs,
                return_attention=True,
            )
        else:
            raw_preds = self.predict(
                smiles_list=smiles_list,
                target_seqs=target_seqs,
                return_attention=False,
            )
            attn_list = None

        pkd_list = []
        kd_list = []
        for val in raw_preds:
            if isinstance(val, list):
                val_scalar = val[0]
            else:
                val_scalar = val
            pkd, kd_nm = self.inverse_transform(val_scalar)
            pkd_list.append(pkd)
            if return_kd_nm:
                kd_list.append(kd_nm)

        res: Dict[str, Any] = {
            "predictions_pkd": pkd_list,
            "raw_predictions": raw_preds,
        }
        if return_kd_nm:
            res["kd_nm"] = kd_list

        if return_attention and attn_list is not None:
            # Convert any numpy arrays in attn_list to lists for JSON serialization
            serialized_attn = []
            for item in attn_list:
                s_item = {}
                for k, v in item.items():
                    s_item[k] = v.tolist() if isinstance(v, np.ndarray) else v
                serialized_attn.append(s_item)
            res["attention_weights"] = serialized_attn

        if return_contact_map and attn_list is not None:
            top_residues_batch = []
            top_atoms_batch = []
            pymol_cmds_batch = []
            contact_maps_batch = []

            for idx, (sm, seq) in enumerate(zip(smiles_list, target_seqs)):
                item_attn = attn_list[idx] if idx < len(attn_list) else {}
                cmap = item_attn.get("contact_map")

                if cmap is not None and isinstance(cmap, np.ndarray) and cmap.ndim == 2:
                    top_res = extract_top_contact_residues(cmap, seq, top_k=top_k_residues)
                    top_at = extract_top_contact_atoms(cmap, sm, top_k=top_k_residues)
                    pymol_cmd = generate_pymol_command(top_res)
                    if return_full_matrix:
                        contact_maps_batch.append(cmap.tolist())
                elif cmap is not None and isinstance(cmap, np.ndarray) and cmap.ndim == 3:
                    # Multi-head attention map: average over heads
                    avg_cmap = np.mean(cmap, axis=0)
                    top_res = extract_top_contact_residues(avg_cmap, seq, top_k=top_k_residues)
                    top_at = extract_top_contact_atoms(avg_cmap, sm, top_k=top_k_residues)
                    pymol_cmd = generate_pymol_command(top_res)
                    if return_full_matrix:
                        contact_maps_batch.append(avg_cmap.tolist())
                else:
                    top_res = []
                    top_at = []
                    pymol_cmd = ""
                    if return_full_matrix:
                        contact_maps_batch.append([])

                top_residues_batch.append(top_res)
                top_atoms_batch.append(top_at)
                pymol_cmds_batch.append(pymol_cmd)

            res["top_contact_residues"] = top_residues_batch
            res["top_contact_atoms"] = top_atoms_batch
            res["pymol_commands"] = pymol_cmds_batch
            if return_full_matrix:
                res["contact_maps"] = contact_maps_batch

        return res


class DTIMultiAffinityPipeline(DTIInferencePipeline):
    """Pipeline specialized for Multi-Affinity (Kd, Ki, IC50) multi-task prediction and ACS evaluation."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cpu",
        aa_max_length: int = 1000,
        scaler_meta: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            model=model,
            device=device,
            aa_max_length=aa_max_length,
            scaler_meta=scaler_meta,
        )
        self.task_names = ["Kd", "Ki", "IC50"]

    def inverse_transform_task(self, y_norm: float, task_idx: int) -> Tuple[float, float]:
        """Convert normalized prediction to pAffinity and affinity in nM for task_idx."""
        task_stats = self.scaler_meta.get("task_stats", {})
        st = None
        if isinstance(task_stats, dict):
            st = task_stats.get(task_idx) or task_stats.get(str(task_idx))

        if st and "mean" in st:
            y_mean = float(st.get("mean", 0.0))
            y_std = float(st.get("std", 1.0))
        elif "y_mean" in self.scaler_meta:
            y_mean = float(self.scaler_meta.get("y_mean", 0.0))
            y_std = float(self.scaler_meta.get("y_std", 1.0))
        else:
            y_mean, y_std = 0.0, 1.0

        log_transform = bool(self.scaler_meta.get("log_transform", True))
        y_log = y_norm * y_std + y_mean
        nm_val = float(np.expm1(y_log)) if log_transform else float(y_log)
        nm_val = max(nm_val, 1e-4)

        p_val = float(9.0 - np.log10(nm_val))
        return round(p_val, 4), round(nm_val, 4)

    def predict_multi_affinity(
        self,
        smiles_list: List[str],
        target_seqs: List[str],
        return_nm: bool = True,
        return_contact_maps: bool = False,
        top_k_residues: int = 10,
        return_full_matrix: bool = False,
    ) -> Dict[str, Any]:
        """Predict Kd, Ki, and IC50 binding affinities simultaneously with Consistency Score."""
        need_attn = return_contact_maps
        if need_attn:
            raw_preds, attn_list = self.predict(
                smiles_list=smiles_list,
                target_seqs=target_seqs,
                return_attention=True,
            )
        else:
            raw_preds = self.predict(
                smiles_list=smiles_list,
                target_seqs=target_seqs,
                return_attention=False,
            )
            attn_list = None

        pkd_list, pki_list, pic50_list = [], [], []
        kd_list, ki_list, ic50_list = [], [], []
        scores_list, tiers_list = [], []

        for val in raw_preds:
            if isinstance(val, (list, tuple)) and len(val) >= 3:
                norm_kd, norm_ki, norm_ic50 = float(val[0]), float(val[1]), float(val[2])
            elif isinstance(val, (list, tuple)) and len(val) == 1:
                norm_kd = norm_ki = norm_ic50 = float(val[0])
            else:
                norm_kd = norm_ki = norm_ic50 = float(val)

            pkd, kd_nm = self.inverse_transform_task(norm_kd, 0)
            pki, ki_nm = self.inverse_transform_task(norm_ki, 1)
            pic50, ic50_nm = self.inverse_transform_task(norm_ic50, 2)

            pkd_list.append(pkd)
            pki_list.append(pki)
            pic50_list.append(pic50)

            if return_nm:
                kd_list.append(kd_nm)
                ki_list.append(ki_nm)
                ic50_list.append(ic50_nm)

            score, tier = compute_affinity_consistency_score(pkd, pki, pic50)
            scores_list.append(score)
            tiers_list.append(tier)

        res: Dict[str, Any] = {
            "predictions_pkd": pkd_list,
            "predictions_pki": pki_list,
            "predictions_pic50": pic50_list,
            "consistency_scores": scores_list,
            "consistency_tiers": tiers_list,
        }
        if return_nm:
            res["kd_nm"] = kd_list
            res["ki_nm"] = ki_list
            res["ic50_nm"] = ic50_list

        if return_contact_maps and attn_list is not None:
            top_residues_batch = []
            top_atoms_batch = []
            pymol_cmds_batch = []
            contact_maps_batch = []

            for idx, (sm, seq) in enumerate(zip(smiles_list, target_seqs)):
                item_attn = attn_list[idx] if idx < len(attn_list) else {}
                cmap = item_attn.get("contact_map")

                if cmap is not None and isinstance(cmap, np.ndarray) and cmap.ndim == 2:
                    top_res = extract_top_contact_residues(cmap, seq, top_k=top_k_residues)
                    top_at = extract_top_contact_atoms(cmap, sm, top_k=top_k_residues)
                    pymol_cmd = generate_pymol_command(top_res)
                    if return_full_matrix:
                        contact_maps_batch.append(cmap.tolist())
                elif cmap is not None and isinstance(cmap, np.ndarray) and cmap.ndim == 3:
                    avg_cmap = np.mean(cmap, axis=0)
                    top_res = extract_top_contact_residues(avg_cmap, seq, top_k=top_k_residues)
                    top_at = extract_top_contact_atoms(avg_cmap, sm, top_k=top_k_residues)
                    pymol_cmd = generate_pymol_command(top_res)
                    if return_full_matrix:
                        contact_maps_batch.append(avg_cmap.tolist())
                else:
                    top_res = []
                    top_at = []
                    pymol_cmd = ""
                    if return_full_matrix:
                        contact_maps_batch.append([])

                top_residues_batch.append(top_res)
                top_atoms_batch.append(top_at)
                pymol_cmds_batch.append(pymol_cmd)

            res["top_contact_residues"] = top_residues_batch
            res["top_contact_atoms"] = top_atoms_batch
            res["pymol_commands"] = pymol_cmds_batch
            if return_full_matrix:
                res["contact_maps"] = contact_maps_batch

        return res


