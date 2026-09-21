"""End-to-End inference pipeline combining molecular featurization and model prediction."""

from typing import Any, List, Optional

import torch

from tdc_studio.data.collate import molecule_collate_fn
from tdc_studio.data.transforms import SequenceTokenizer, SmilesToGraphTransform


class InferencePipeline:
    """Unified inference engine encapsulating CPU featurization and PyTorch model execution."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cpu",
        is_dta: bool = False,
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.is_dta = is_dta
        self.graph_transform = SmilesToGraphTransform()
        self.seq_tokenizer = SequenceTokenizer() if is_dta else None

    def predict(
        self, smiles_list: List[str], target_seqs: Optional[List[str]] = None
    ) -> List[float]:
        """Run batch inference on raw SMILES and optional targets."""
        batch_items = []
        for i, sm in enumerate(smiles_list):
            g = self.graph_transform(sm)
            if g is None:
                # If invalid SMILES, create dummy graph with 0-filled feature
                from torch_geometric.data import Data

                g = Data(x=torch.zeros((1, 14)), edge_index=torch.empty((2, 0), dtype=torch.long))

            item: dict[str, Any] = {"drug_graph": g}
            if self.is_dta and target_seqs is not None and i < len(target_seqs):
                item["target_seq"] = self.seq_tokenizer(target_seqs[i])
            batch_items.append(item)

        # Collate items
        collated = molecule_collate_fn(batch_items)

        # Move to device
        collated["drug_graph"] = collated["drug_graph"].to(self.device)
        if "target_seq" in collated:
            collated["target_seq"] = collated["target_seq"].to(self.device)

        with torch.no_grad():
            preds = self.model(collated)
            preds_flat = preds.squeeze(-1).detach().cpu().numpy().tolist()

        if isinstance(preds_flat, float):
            return [preds_flat]
        return preds_flat
