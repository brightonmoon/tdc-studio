"""Directed Message Passing Neural Network (D-MPNN) with optional 2D Descriptors."""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.utils import scatter

from tdc_studio.core.registry import MODELS
from tdc_studio.models.base import BaseTherapeuticsModel
from tdc_studio.models.loss.multitask_loss import MaskedMultiTaskLoss


@MODELS.register("dmpnn")
@MODELS.register("dmpnn_des")
@MODELS.register("dmpnn_mtl")
class DMPNNModel(BaseTherapeuticsModel):
    """Directed Message Passing Neural Network (Chemprop / Yang et al. 2019).

    Operates on directed bonds e_{vw} rather than atoms, eliminating backtracking loops.
    Optionally incorporates 210 RDKit 2D physico-chemical descriptors via a hybrid MLP branch.
    Supports single-task regression/classification and multi-task learning with task-specific heads.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        in_dim = config.get("in_dim", 14)  # Atom features
        edge_dim = config.get("edge_dim", 6)  # Bond features
        hidden_dim = config.get("hidden_dim", 300)
        depth = config.get("depth", config.get("num_layers", 3))
        dropout = config.get("dropout", 0.15)
        self.depth = depth

        # Initial directed bond projection: [x_u || e_{uv}] -> hidden_dim
        self.w_i = nn.Linear(in_dim + edge_dim, hidden_dim, bias=False)

        # Message passing transformation: hidden_dim -> hidden_dim
        self.w_m = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Atom readout projection: [x_v || incoming_edges_sum] -> hidden_dim
        self.w_a = nn.Linear(in_dim + hidden_dim, hidden_dim)

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Optional RDKit 2D Descriptors Branch (DMPNN-Des)
        self.use_descriptors = config.get("use_descriptors", True)
        self.descriptor_dim = config.get("descriptor_dim", 210)

        # Representation combination: Mean pool + Add pool = 2 * hidden_dim
        pooled_dim = hidden_dim * 2

        if self.use_descriptors:
            self.desc_encoder = nn.Sequential(
                nn.BatchNorm1d(self.descriptor_dim),
                nn.Linear(self.descriptor_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            head_in_dim = pooled_dim + hidden_dim
        else:
            head_in_dim = pooled_dim

        # Multi-task heads setup (if tasks specified in config)
        raw_tasks = config.get("tasks", None)
        if raw_tasks:
            self.task_names = [t["name"] if isinstance(t, dict) else t for t in raw_tasks]
            self.task_types = [
                t.get("type", "regression") if isinstance(t, dict) else "regression"
                for t in raw_tasks
            ]
            self.num_tasks = len(self.task_names)
            self.task_heads = nn.ModuleDict(
                {
                    name: nn.Sequential(
                        nn.Linear(head_in_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_dim, 1),
                    )
                    for name in self.task_names
                }
            )
            use_uncertainty = config.get("use_uncertainty", False)
            task_weights = config.get("task_weights", None)
            pearson_weight = config.get("pearson_weight", 0.0)
            r2_weight = config.get("r2_weight", 0.0)
            self.loss_fn = MaskedMultiTaskLoss(
                task_names=self.task_names,
                task_types=self.task_types,
                use_uncertainty=use_uncertainty,
                task_weights=task_weights,
                pearson_weight=pearson_weight,
                r2_weight=r2_weight,
            )
            self.head = None
        else:
            self.task_names = []
            self.num_tasks = 1
            self.task_heads = None
            self.loss_fn = None
            # Standard Chemprop-style 2-layer FFN Head
            self.head = nn.Sequential(
                nn.Linear(head_in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

    def _extract_graph_features(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Extract graph molecular representation via directed message passing."""
        graph = batch["drug_graph"]
        x, edge_index, batch_idx = graph.x, graph.edge_index, graph.batch
        num_nodes = x.size(0)
        num_edges = edge_index.size(1)

        if num_edges == 0:
            # Molecules with no bonds (single atom edge case)
            h_v = self.act(
                self.w_a(
                    torch.cat(
                        [x, torch.zeros(num_nodes, self.w_m.out_features, device=x.device)], dim=-1
                    )
                )
            )
            h_mean = global_mean_pool(h_v, batch_idx)
            h_sum = global_add_pool(h_v, batch_idx)
            return torch.cat([h_mean, h_sum], dim=-1)

        src, dst = edge_index[0], edge_index[1]
        edge_attr = (
            graph.edge_attr
            if (
                hasattr(graph, "edge_attr")
                and graph.edge_attr is not None
                and graph.edge_attr.numel() > 0
            )
            else torch.zeros((num_edges, 6), device=x.device)
        )

        # 1. Initial edge representation h^(0)
        init_input = torch.cat([x[src], edge_attr], dim=-1)
        h0 = self.act(self.w_i(init_input))
        h = h0

        # Reverse edge indices (pairs in SmilesToGraphTransform)
        rev = torch.arange(num_edges, device=edge_index.device) ^ 1

        # 2. Directed Message Passing Loop
        for _ in range(self.depth):
            # Sum of incoming edge states to each node v
            incoming_sum = scatter(h, dst, dim=0, dim_size=num_nodes, reduce="sum")
            # Message along edge k=(u -> v): sum of incoming to u minus reverse edge (v -> u)
            msg = incoming_sum[src] - h[rev]
            h = self.act(h0 + self.w_m(msg))
            h = self.dropout(h)

        # 3. Atom Readout: sum of incoming edges to atom v
        node_incoming = scatter(h, dst, dim=0, dim_size=num_nodes, reduce="sum")
        atom_rep = self.act(self.w_a(torch.cat([x, node_incoming], dim=-1)))
        atom_rep = self.dropout(atom_rep)

        # 4. Molecule Readout: dual mean + sum pooling
        h_mean = global_mean_pool(atom_rep, batch_idx)
        h_sum = global_add_pool(atom_rep, batch_idx)
        return torch.cat([h_mean, h_sum], dim=-1)

    def extract_features(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Extract combined molecular representation (Graph + optional Descriptors)."""
        h_mol = self._extract_graph_features(batch)

        if (
            self.use_descriptors
            and hasattr(self, "desc_encoder")
            and "descriptors" in batch
            and batch["descriptors"] is not None
        ):
            desc = batch["descriptors"]
            if desc.size(0) == 1 and self.desc_encoder[0].training:
                h_desc = self.desc_encoder[1:](desc)
            else:
                h_desc = self.desc_encoder(desc)
            return torch.cat([h_mol, h_desc], dim=-1)

        return h_mol

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Forward pass generating predictions for single or multiple tasks."""
        h = self.extract_features(batch)

        if hasattr(self, "task_heads") and self.task_heads is not None:
            preds = [self.task_heads[name](h).squeeze(-1) for name in self.task_names]
            return torch.stack(preds, dim=-1)

        return self.head(h)

    def compute_loss(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute training loss (masked multi-task loss or single-task loss)."""
        if hasattr(self, "loss_fn") and self.loss_fn is not None:
            loss, _ = self.loss_fn(preds, targets, mask=mask)
            return loss
        return super().compute_loss(preds, targets)
