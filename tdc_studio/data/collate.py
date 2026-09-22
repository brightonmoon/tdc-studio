"""Custom collate functions for PyTorch Geometric graphs, sequences, and fingerprints."""

from typing import Any, Dict, List

import torch
from torch_geometric.data import Batch, Data


def molecule_collate_fn(batch_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate list of samples into a unified PyTorch batch across multiple modalities.

    Supported keys per sample:
      - 'drug_graph': PyG Data -> PyG Batch (with edge_attr preserved)
      - 'smiles_seq': 1D Tensor -> 2D Padded LongTensor [B, max_len]
      - 'fingerprint': 1D FloatTensor -> 2D Stacked FloatTensor [B, n_bits]
      - 'target_seq': 1D Tensor -> 2D Padded LongTensor [B, max_target_len]
      - 'label': scalar float -> 1D FloatTensor [B]
    """
    valid_items = [
        it
        for it in batch_items
        if any(
            it.get(k) is not None for k in ("drug_graph", "smiles_seq", "fingerprint", "target_seq")
        )
    ]
    if not valid_items:
        raise ValueError("All samples in batch have invalid molecular representations.")

    batch: Dict[str, Any] = {}

    # 1. Collate drug graphs using PyTorch Geometric Batch
    if "drug_graph" in valid_items[0] and valid_items[0]["drug_graph"] is not None:
        graph_list: List[Data] = [item["drug_graph"] for item in valid_items]
        batch["drug_graph"] = Batch.from_data_list(graph_list)

    # 2. Collate SMILES token sequences if present (padding with 0)
    if "smiles_seq" in valid_items[0] and valid_items[0]["smiles_seq"] is not None:
        smiles_seqs = [item["smiles_seq"] for item in valid_items]
        batch["smiles_seq"] = torch.nn.utils.rnn.pad_sequence(
            smiles_seqs, batch_first=True, padding_value=0
        )

    # 3. Collate Morgan fingerprints if present (stacked tensor)
    if "fingerprint" in valid_items[0] and valid_items[0]["fingerprint"] is not None:
        fps = [item["fingerprint"] for item in valid_items]
        batch["fingerprint"] = torch.stack(fps, dim=0)

    # 4. Collate target sequences if present (padding with 0)
    if "target_seq" in valid_items[0] and valid_items[0]["target_seq"] is not None:
        seqs = [item["target_seq"] for item in valid_items]
        batch["target_seq"] = torch.nn.utils.rnn.pad_sequence(
            seqs, batch_first=True, padding_value=0
        )

    # 5. Collate labels if present (single scalar or multi-task vector)
    if "label" in valid_items[0]:
        labels = [item["label"] for item in valid_items]
        batch["labels"] = torch.tensor(labels, dtype=torch.float32)
    elif "labels" in valid_items[0]:
        labels = [item["labels"] for item in valid_items]
        batch["labels"] = (
            torch.stack(labels, dim=0)
            if isinstance(labels[0], torch.Tensor)
            else torch.tensor(labels, dtype=torch.float32)
        )

    # 6. Collate task mask if present (for missing labels)
    if "mask" in valid_items[0]:
        masks = [item["mask"] for item in valid_items]
        batch["mask"] = (
            torch.stack(masks, dim=0)
            if isinstance(masks[0], torch.Tensor)
            else torch.tensor(masks, dtype=torch.bool)
        )

    return batch
