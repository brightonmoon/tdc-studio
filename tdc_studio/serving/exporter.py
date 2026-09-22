"""Model export utilities for production deployment."""

import json
import os
import shutil
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import torch

from tdc_studio.core.exceptions import ServingError
from tdc_studio.core.registry import MODELS


def export_model_checkpoint(
    model: torch.nn.Module,
    model_config: Dict[str, Any],
    output_dir: str,
    checkpoint_name: str = "model.pt",
    extra_meta: Optional[Dict[str, Any]] = None,
) -> str:
    """Export model weights, architecture config, and optional metadata into an artifact folder."""
    os.makedirs(output_dir, exist_ok=True)

    weights_path = os.path.join(output_dir, checkpoint_name)
    torch.save(model.state_dict(), weights_path)

    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=2)

    if extra_meta:
        meta_path = os.path.join(output_dir, "training_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(extra_meta, f, indent=2)

    return weights_path


def load_model_from_checkpoint(
    checkpoint_dir: str,
    model_cls: Optional[Any] = None,
    weights_name: Optional[str] = None,
) -> torch.nn.Module:
    """Load model from exported checkpoint folder with auto-detection of model class and weights."""
    config_path = os.path.join(checkpoint_dir, "config.json")
    if not os.path.exists(config_path):
        raise ServingError(f"Model configuration file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Auto-resolve model class if not provided
    if model_cls is None:
        model_type = config.get("type")
        if not model_type:
            raise ServingError("config.json does not specify 'type' field for model architecture.")
        model_cls = MODELS.get(model_type)

    # Auto-detect weights file
    if weights_name:
        weights_path = os.path.join(checkpoint_dir, weights_name)
    else:
        candidates = ["best_model.pt", "model.pt"]
        weights_path = None
        for cand in candidates:
            p = os.path.join(checkpoint_dir, cand)
            if os.path.exists(p):
                weights_path = p
                break
        if weights_path is None:
            raise ServingError(
                f"No weights file ({candidates}) found in checkpoint dir: {checkpoint_dir}"
            )

    model = model_cls(config)
    try:
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    except TypeError:
        # Fallback for older torch versions without weights_only param
        state_dict = torch.load(weights_path, map_location="cpu")

    model.load_state_dict(state_dict)
    model.eval()
    return model


def export_production_package(
    checkpoint_dir: str,
    output_dir: str,
    checkpoint_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate, package, and export model artifacts for production serving."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load and validate model from checkpoint
    model = load_model_from_checkpoint(checkpoint_dir, weights_name=checkpoint_name)

    # 2. Read existing config
    config_path = os.path.join(checkpoint_dir, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # 3. Copy/save weights to output_dir as standardized "model.pt"
    dest_weights = os.path.join(output_dir, "model.pt")
    torch.save(model.state_dict(), dest_weights)

    # 4. Save standardized config.json
    dest_config = os.path.join(output_dir, "config.json")
    with open(dest_config, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # 5. Copy training_meta.json if present
    src_meta = os.path.join(checkpoint_dir, "training_meta.json")
    training_meta = {}
    if os.path.exists(src_meta):
        with open(src_meta, "r", encoding="utf-8") as f:
            training_meta = json.load(f)
        shutil.copyfile(src_meta, os.path.join(output_dir, "training_meta.json"))

    # 6. Generate export manifest
    param_count = sum(p.numel() for p in model.parameters())
    manifest = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "model_type": config.get("type"),
        "task_type": config.get("task_type", "regression"),
        "parameter_count": param_count,
        "source_checkpoint": os.path.abspath(checkpoint_dir),
        "weights_file": "model.pt",
        "config_file": "config.json",
        "training_meta": training_meta,
    }
    manifest_path = os.path.join(output_dir, "export_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest
