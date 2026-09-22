"""3-Tier Hybrid Categorical Multi-Task Learning (MTL) Architecture for ADMET prediction."""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from tdc_studio.core.registry import MODELS
from tdc_studio.models.base import BaseTherapeuticsModel
from tdc_studio.models.loss.multitask_loss import MaskedMultiTaskLoss


@MODELS.register("categorical_mtl")

class CategoricalMTLModel(BaseTherapeuticsModel):
    """3-Tier Hybrid Architecture for ADMET prediction.

    - Tier 1: Foundation / Generic Molecular Backbone (Frozen or fine-tuned)
    - Tier 2: Categorical Multi-Task Heads (Absorption, Distribution, Metabolism, Excretion, Toxicity, etc.)
    - Tier 3: Specialized Single Adapters for high-sensitivity endpoints (Caco-2, hERG)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # 1. Parse Task Configurations
        raw_tasks = config.get("tasks", [])
        self.task_configs: List[Dict[str, str]] = []
        for t in raw_tasks:
            if isinstance(t, str):
                self.task_configs.append({"name": t, "category": "general", "type": "regression"})
            elif isinstance(t, dict):
                self.task_configs.append(
                    {
                        "name": t["name"],
                        "category": t.get("category", "general"),
                        "type": t.get("type", "regression"),
                    }
                )

        if not self.task_configs:
            # Fallback default 3-task setup
            self.task_configs = [
                {"name": "caco2_wang", "category": "absorption", "type": "regression"},
                {"name": "hia_hou", "category": "absorption", "type": "binary_classification"},
                {"name": "herg", "category": "toxicity", "type": "binary_classification"},
            ]

        self.task_names = [t["name"] for t in self.task_configs]
        self.task_types = [t["type"] for t in self.task_configs]
        self.task_categories = [t["category"] for t in self.task_configs]
        self.num_tasks = len(self.task_configs)
        self.task_name_to_idx = {name: i for i, name in enumerate(self.task_names)}

        # Group tasks by category
        self.category_groups: Dict[str, List[str]] = {}
        for t in self.task_configs:
            self.category_groups.setdefault(t["category"], []).append(t["name"])

        # 2. Tier 1: Backbone Setup & Freezing
        self.freeze_backbone = config.get("freeze_backbone", False)
        backbone_type = config.get("backbone_type", "gine")
        backbone_kwargs = config.get("backbone_config", {})

        # Default hidden dim depending on backbone type
        default_backbone_dim = 256 if backbone_type == "gine" else (512 if backbone_type == "mlp_baseline" else 128)
        self.backbone_hidden_dim = config.get("backbone_hidden_dim", default_backbone_dim)

        if "backbone_instance" in config:
            self.backbone = config["backbone_instance"]
        else:
            backbone_cls = MODELS.get(backbone_type)
            self.backbone = backbone_cls(backbone_kwargs)

        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # 3. Tier 2: Categorical Multi-Task Heads
        head_hidden_dim = config.get("head_hidden_dim", 128)
        dropout = config.get("dropout", 0.1)

        self.category_heads = nn.ModuleDict()
        for cat_name, task_list in self.category_groups.items():
            k_tasks = len(task_list)
            self.category_heads[cat_name] = nn.Sequential(
                nn.Linear(self.backbone_hidden_dim, head_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden_dim, k_tasks),
            )

        # 4. Tier 3: Specialized Single Adapters for High-Sensitivity Endpoints
        specialized_tasks = config.get("specialized_tasks", ["caco2_wang", "herg"])
        self.specialized_tasks = [t for t in specialized_tasks if t in self.task_name_to_idx]

        self.specialized_adapters = nn.ModuleDict()
        self.adapter_scales = nn.ParameterDict()
        for s_task in self.specialized_tasks:
            # 2-layer residual adapter
            self.specialized_adapters[s_task] = nn.Sequential(
                nn.Linear(self.backbone_hidden_dim, head_hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden_dim // 2, 1),
            )
            # Learnable scale initialized to 0.1
            self.adapter_scales[s_task] = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

        # 5. Masked Multi-Task Loss with Homoscedastic Uncertainty
        use_uncertainty = config.get("use_uncertainty", True)
        self.loss_fn = MaskedMultiTaskLoss(
            task_names=self.task_names,
            task_types=self.task_types,
            use_uncertainty=use_uncertainty,
        )

    def compute_loss(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute masked multi-task loss with uncertainty weighting."""
        loss, _ = self.loss_fn(preds, targets, mask)
        return loss

    def extract_features(self, batch: Dict[str, Any]) -> torch.Tensor:

        """Extract molecular representation from Tier 1 backbone."""
        if self.freeze_backbone:
            with torch.no_grad():
                if hasattr(self.backbone, "extract_features"):
                    h = self.backbone.extract_features(batch)
                else:
                    h = self.backbone(batch)
        else:
            if hasattr(self.backbone, "extract_features"):
                h = self.backbone.extract_features(batch)
            else:
                h = self.backbone(batch)

        if h.ndim == 1:
            h = h.unsqueeze(0)
        return h

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Forward pass generating predictions for all tasks [B, num_tasks]."""
        h = self.extract_features(batch)
        b_size = h.size(0)
        device = h.device

        # Preallocate output predictions tensor [B, num_tasks]
        preds = torch.zeros((b_size, self.num_tasks), dtype=torch.float32, device=device)

        # Compute Tier 2 category heads
        for cat_name, task_list in self.category_groups.items():
            head = self.category_heads[cat_name]
            cat_out = head(h)  # [B, len(task_list)]

            for local_idx, t_name in enumerate(task_list):
                global_idx = self.task_name_to_idx[t_name]
                t_pred = cat_out[:, local_idx]

                # Tier 3: Apply specialized adapter if present
                if t_name in self.specialized_adapters:
                    adapter = self.specialized_adapters[t_name]
                    scale = self.adapter_scales[t_name]
                    residual = adapter(h).squeeze(-1)
                    t_pred = t_pred + scale * residual

                preds[:, global_idx] = t_pred

        return preds

    def predict_dict(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Convenience method returning predictions mapped to task names."""
        preds = self.forward(batch)
        return {name: preds[:, i] for i, name in enumerate(self.task_names)}

    def parameter_summary(self) -> Dict[str, int]:
        """Summarize trainable parameter count across tiers."""
        backbone_params = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        frozen_backbone_params = sum(p.numel() for p in self.backbone.parameters() if not p.requires_grad)
        head_params = sum(p.numel() for p in self.category_heads.parameters() if p.requires_grad)
        adapter_params = sum(p.numel() for p in self.specialized_adapters.parameters() if p.requires_grad) + len(
            self.adapter_scales
        )
        return {
            "trainable_backbone": backbone_params,
            "frozen_backbone": frozen_backbone_params,
            "category_heads": head_params,
            "specialized_adapters": adapter_params,
            "total_trainable": backbone_params + head_params + adapter_params,
        }
