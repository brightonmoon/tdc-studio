"""Multi-task inference engine and medicinal chemistry decision thresholds."""

import json
import os
import time
from typing import Any, Dict, List

import torch
from torch_geometric.data import Data

from tdc_studio.core.registry import MODELS
from tdc_studio.data.collate import molecule_collate_fn
from tdc_studio.data.transforms import (
    CanonicalSmilesNormalizer,
    MorganFingerprintTransform,
    SmilesToGraphTransform,
    SmilesTokenizer,
)


class ThresholdDecisionEngine:
    """Medicinal chemistry decision rule engine mapping predictions to actionable tiers."""

    @staticmethod
    def evaluate(task_name: str, raw_pred: float, task_type: str) -> Dict[str, Any]:
        """Convert raw prediction/logit into value/probability and qualitative decision."""
        clean_name = task_name.lower()
        is_classification = task_type in ("binary_classification", "classification")

        if is_classification:
            # Convert logit to probability if necessary
            if raw_pred < 0.0 or raw_pred > 1.0:
                prob = float(torch.sigmoid(torch.tensor(raw_pred)).item())
            else:
                prob = float(raw_pred)

            prob_rounded = round(prob, 4)

            # Endpoint specific classification rules
            if clean_name == "herg":
                if prob < 0.3:
                    decision = "Low Cardiotoxicity Risk"
                elif prob < 0.7:
                    decision = "Moderate Cardiotoxicity Risk"
                else:
                    decision = "High Cardiotoxicity Risk (hERG Blocker)"
                return {"probability": prob_rounded, "decision": decision, "unit": "probability"}

            if clean_name == "dili":
                decision = "Low Hepatotoxicity Risk (Safe)" if prob < 0.5 else "High Hepatotoxicity Risk (DILI+)"
                return {"probability": prob_rounded, "decision": decision, "unit": "probability"}

            if clean_name == "ames":
                decision = "Non-Mutagenic (Safe)" if prob < 0.5 else "Mutagenic (AMES+)"
                return {"probability": prob_rounded, "decision": decision, "unit": "probability"}

            if clean_name in ("hia_hou", "bioavailability_ma"):
                decision = "High Absorption" if prob >= 0.5 else "Low Absorption"
                return {"probability": prob_rounded, "decision": decision, "unit": "probability"}

            if clean_name == "bbb_martins":
                decision = "BBB Penetrant (BBB+)" if prob >= 0.5 else "Non-Penetrant (BBB-)"
                return {"probability": prob_rounded, "decision": decision, "unit": "probability"}

            # Generic classification
            decision = "Positive" if prob >= 0.5 else "Negative"
            return {"probability": prob_rounded, "decision": decision, "unit": "probability"}

        # Regression endpoints
        val = round(float(raw_pred), 4)

        if clean_name == "caco2_wang":
            # Caco-2 permeability (log Papp, 10^-6 cm/s)
            if val > -5.15:
                decision = "High Permeability (> -5.15)"
            elif val >= -6.0:
                decision = "Moderate Permeability (-6.0 to -5.15)"
            else:
                decision = "Low Permeability (< -6.0)"
            return {"value": val, "unit": "10^-6 cm/s", "decision": decision}

        if clean_name == "ppbr_az":
            # Plasma protein binding rate (%)
            if val <= 80.0:
                decision = "Low Binding (<= 80%)"
            elif val <= 95.0:
                decision = "Moderate Binding (80 - 95%)"
            else:
                decision = "High Binding (> 95%)"
            return {"value": val, "unit": "%", "decision": decision}

        if clean_name == "solubility_aqsoldb":
            # Aqueous solubility (log mol/L)
            if val >= -2.0:
                decision = "High Solubility"
            elif val >= -4.0:
                decision = "Moderate Solubility"
            else:
                decision = "Low / Insoluble"
            return {"value": val, "unit": "log mol/L", "decision": decision}

        if clean_name == "lipophilicity_astrazeneca":
            # Lipophilicity logD
            if 1.0 <= val <= 3.0:
                decision = "Optimal Lipophilicity (Lipinski Rule)"
            elif val < 1.0:
                decision = "Hydrophilic"
            else:
                decision = "High Lipophilicity"
            return {"value": val, "unit": "logD", "decision": decision}

        # Generic regression
        return {"value": val, "unit": "arbitrary", "decision": f"{val:.4f}"}


class MultiTaskInferencePipeline:
    """Production inference pipeline evaluating multi-task ADMET predictions and decision tiers."""

    def __init__(
        self,
        model: torch.nn.Module,
        task_configs: List[Dict[str, str]],
        modality: str = "graph",
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.model.eval()
        self.task_configs = task_configs
        self.task_names = [t["name"] for t in task_configs]
        self.task_types = [t.get("type", "regression") for t in task_configs]
        self.modality = modality.lower()
        self.device = device

        self.normalizer = CanonicalSmilesNormalizer(remove_salts=True)
        self.graph_transform = SmilesToGraphTransform()
        self.smiles_tokenizer = SmilesTokenizer()
        self.fingerprint_transform = MorganFingerprintTransform()
        self.decision_engine = ThresholdDecisionEngine()

    @classmethod
    def from_pretrained(
        cls, model_dir: str, device: str = "cpu"
    ) -> "MultiTaskInferencePipeline":
        """Load trained multi-task pipeline from exported directory."""
        config_path = os.path.join(model_dir, "config.json")
        weights_path = os.path.join(model_dir, "model.pt")

        if not os.path.exists(config_path) or not os.path.exists(weights_path):
            raise FileNotFoundError(f"Missing config.json or model.pt in {model_dir}")

        with open(config_path, "r", encoding="utf-8") as f:
            full_config = json.load(f)

        model_name = full_config.get("model", {}).get("name", "categorical_mtl")
        model_cfg = full_config.get("model", {})
        data_cfg = full_config.get("data", {})

        model_cls = MODELS.get(model_name)
        model = model_cls(model_cfg)
        model.load_state_dict(torch.load(weights_path, map_location=device))

        tasks = model_cfg.get("tasks", data_cfg.get("tasks", []))
        modality = data_cfg.get("modality", "graph")

        return cls(model=model, task_configs=tasks, modality=modality, device=device)

    def predict(self, smiles: str) -> Dict[str, Any]:
        """Predict ADMET profile for a single SMILES string with decision labels."""
        start_t = time.perf_counter()

        # 1. SMILES normalization
        canon_s = self.normalizer(smiles) or smiles

        # 2. Featurize
        item: Dict[str, Any] = {}
        if self.modality == "sequence":
            item["smiles_seq"] = self.smiles_tokenizer(canon_s)
        elif self.modality == "fingerprint":
            fp = self.fingerprint_transform(canon_s)
            if fp is None:
                fp = torch.zeros(2048, dtype=torch.float32)
            item["fingerprint"] = fp
        else:
            g = self.graph_transform(canon_s)
            if g is None:
                g = Data(
                    x=torch.zeros((1, 14)),
                    edge_index=torch.empty((2, 0), dtype=torch.long),
                    edge_attr=torch.empty((0, 6), dtype=torch.float),
                )
            item["drug_graph"] = g

        collated = molecule_collate_fn([item])
        for k, v in collated.items():
            if hasattr(v, "to"):
                collated[k] = v.to(self.device)

        # 3. Model forward
        with torch.no_grad():
            preds = self.model(collated)  # [1, num_tasks]
            preds_row = preds[0].cpu().numpy()

        # 4. Map decisions
        decisions: Dict[str, Any] = {}
        for idx, t_name in enumerate(self.task_names):
            t_type = self.task_types[idx]
            raw_val = float(preds_row[idx])
            decisions[t_name] = self.decision_engine.evaluate(t_name, raw_val, t_type)

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0

        return {
            "smiles": canon_s,
            "raw_smiles": smiles,
            "elapsed_ms": round(elapsed_ms, 2),
            "predictions": decisions,
        }

    def predict_batch(self, smiles_list: List[str]) -> List[Dict[str, Any]]:
        """Run batch inference for multiple SMILES strings."""
        return [self.predict(s) for s in smiles_list]
