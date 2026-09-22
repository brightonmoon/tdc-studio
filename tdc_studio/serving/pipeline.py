"""End-to-End inference pipeline combining molecular featurization and model prediction."""

from typing import Any, List, Optional

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
