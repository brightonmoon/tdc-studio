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
    ) -> List[float]:
        """Run DTI affinity prediction for paired (SMILES, AA sequence) inputs.

        Args:
            smiles_list : List of drug SMILES strings.
            target_seqs : List of protein AA sequences (same length as smiles_list).

        Returns:
            List of raw model output predictions.
        """
        if target_seqs is None or len(target_seqs) != len(smiles_list):
            raise ValueError(
                "DTIInferencePipeline requires target_seqs of the same length as smiles_list."
            )

        batch_items = []
        for sm, seq in zip(smiles_list, target_seqs):
            clean_smiles = sm.strip()
            clean_seq = seq.strip().upper()

            # Drug graph
            g = self.graph_transform(clean_smiles)
            if g is None:
                g = Data(
                    x=torch.zeros((1, 14)),
                    edge_index=torch.empty((2, 0), dtype=torch.long),
                    edge_attr=torch.empty((0, 6), dtype=torch.float),
                )

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

        with torch.no_grad():
            preds = self.model(collated)
            preds_flat = preds.squeeze(-1).detach().cpu().numpy().tolist()

        if isinstance(preds_flat, float):
            return [preds_flat]
        return preds_flat

    def predict_affinity(
        self,
        smiles_list: List[str],
        target_seqs: List[str],
        return_kd_nm: bool = True,
    ) -> Dict[str, Any]:
        """Run DTI prediction and return structured pKd and Kd (nM) values.

        Args:
            smiles_list : List of drug SMILES strings.
            target_seqs : List of protein AA sequences.
            return_kd_nm: Whether to compute Kd in nM.

        Returns:
            Dict containing 'predictions_pkd', 'kd_nm' (optional), and 'raw_predictions'.
        """
        raw_preds = self.predict(smiles_list=smiles_list, target_seqs=target_seqs)

        pkd_list = []
        kd_list = []
        for val in raw_preds:
            pkd, kd_nm = self.inverse_transform(val)
            pkd_list.append(pkd)
            if return_kd_nm:
                kd_list.append(kd_nm)

        res: Dict[str, Any] = {
            "predictions_pkd": pkd_list,
            "raw_predictions": raw_preds,
        }
        if return_kd_nm:
            res["kd_nm"] = kd_list
        return res


