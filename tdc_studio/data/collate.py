"""Custom collate functions for PyTorch Geometric batches and variable-length sequences."""

from typing import Any, Dict, List

import torch
from torch_geometric.data import Batch, Data


def molecule_collate_fn(batch_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate list of samples into a PyTorch batch.

    Supports:
      - 'drug_graph': PyG Data -> PyG Batch
      - 'target_seq': 1D Tensor -> 2D Padded Tensor
      - 'label': scalar float -> 1D Tensor
    """
    valid_items = [it for it in batch_items if it.get("drug_graph") is not None]
    if not valid_items:
        raise ValueError("All samples in batch have invalid drug_graph representations.")

    batch: Dict[str, Any] = {}

    # 1. Collate drug graphs using PyTorch Geometric Batch
    graph_list: List[Data] = [item["drug_graph"] for item in valid_items]
    batch["drug_graph"] = Batch.from_data_list(graph_list)

    # 2. Collate target sequences if present (padding)
    if "target_seq" in valid_items[0]:
        seqs = [item["target_seq"] for item in valid_items]
        batch["target_seq"] = torch.nn.utils.rnn.pad_sequence(
            seqs, batch_first=True, padding_value=0
        )

    # 3. Collate labels if present
    if "label" in valid_items[0]:
        labels = [item["label"] for item in valid_items]
        batch["labels"] = torch.tensor(labels, dtype=torch.float32)

    return batch
