"""Model export utilities for production deployment."""

import json
import os
from typing import Any, Dict

import torch


def export_model_checkpoint(
    model: torch.nn.Module,
    model_config: Dict[str, Any],
    output_dir: str,
    checkpoint_name: str = "model.pt",
) -> str:
    """Export model weights and architecture config into an artifact folder."""
    os.makedirs(output_dir, exist_ok=True)

    weights_path = os.path.join(output_dir, checkpoint_name)
    torch.save(model.state_dict(), weights_path)

    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(model_config, f, indent=2)

    return weights_path


def load_model_from_checkpoint(checkpoint_dir: str, model_cls: Any) -> torch.nn.Module:
    """Load model from exported checkpoint folder."""
    config_path = os.path.join(checkpoint_dir, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    model = model_cls(config)
    weights_path = os.path.join(checkpoint_dir, "model.pt")
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model
