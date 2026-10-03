"""Model export utilities for production deployment."""

import json
import os
import shutil
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

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
    """Load model from exported checkpoint folder with auto-detection of model class and weights.

    Supports both raw state_dict checkpoints and wrapped training dict checkpoints
    (e.g., {"model_state_dict": ..., "config": ...}).
    """
    config_path = os.path.join(checkpoint_dir, "config.json")
    config: Dict[str, Any] = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

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

    try:
        raw_loaded = torch.load(weights_path, map_location="cpu", weights_only=True)
    except TypeError:
        # Fallback for older torch versions without weights_only param
        raw_loaded = torch.load(weights_path, map_location="cpu")

    # If config was not found on disk, attempt extracting from checkpoint payload
    if not config and isinstance(raw_loaded, dict) and "config" in raw_loaded:
        embedded_cfg = raw_loaded["config"]
        if isinstance(embedded_cfg, dict):
            config = embedded_cfg.get("model", embedded_cfg)

    if not config:
        raise ServingError(
            f"Model configuration not found in {checkpoint_dir} or inside checkpoint weights."
        )

    # Auto-resolve model class if not provided
    if model_cls is None:
        model_type = config.get("type")
        if not model_type:
            raise ServingError("Model config does not specify 'type' field for model architecture.")
        model_cls = MODELS.get(model_type)

    # Unwrap state_dict if checkpoint was saved as training dictionary
    if isinstance(raw_loaded, dict) and "model_state_dict" in raw_loaded:
        state_dict = raw_loaded["model_state_dict"]
    elif isinstance(raw_loaded, dict) and "state_dict" in raw_loaded:
        state_dict = raw_loaded["state_dict"]
    else:
        state_dict = raw_loaded

    model = model_cls(config)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        # Fallback to strict=False for cross-environment transformers rotary buffer differences
        model.load_state_dict(state_dict, strict=False)
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

    # 5. Copy training_meta.json and scaler.json if present
    src_meta = os.path.join(checkpoint_dir, "training_meta.json")
    training_meta = {}
    if os.path.exists(src_meta):
        with open(src_meta, "r", encoding="utf-8") as f:
            training_meta = json.load(f)
        shutil.copyfile(src_meta, os.path.join(output_dir, "training_meta.json"))

    src_scaler = os.path.join(checkpoint_dir, "scaler.json")
    has_scaler = False
    if os.path.exists(src_scaler):
        shutil.copyfile(src_scaler, os.path.join(output_dir, "scaler.json"))
        has_scaler = True

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
        "has_scaler": has_scaler,
        "training_meta": training_meta,
    }
    manifest_path = os.path.join(output_dir, "export_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def export_tri_hybrid_package(
    dmpnn_checkpoint_path: str,
    dmpnn_config_path: str,
    gbdt_model: Any,
    ridge_model: Any,
    scaler: Any,
    blending_weights: Tuple[float, float, float],
    calibration_params: Tuple[float, float],
    dmpnn_stats: Tuple[float, float],
    output_dir: str,
    benchmark_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Package and export full Tri-Hybrid SOTA model into production serving directory."""
    import pickle

    os.makedirs(output_dir, exist_ok=True)

    dmpnn_ckpt = torch.load(dmpnn_checkpoint_path, map_location="cpu", weights_only=False)
    with open(dmpnn_config_path, "r", encoding="utf-8") as f:
        dmpnn_cfg = json.load(f)

    # Package all components into single .pt file
    package = {
        "dmpnn_state_dict": dmpnn_ckpt,
        "dmpnn_config": dmpnn_cfg,
        "gbdt_pickle": pickle.dumps(gbdt_model),
        "ridge_pickle": pickle.dumps(ridge_model),
        "scaler_pickle": pickle.dumps(scaler),
        "blending_weights": blending_weights,
        "calibration_params": calibration_params,
        "dmpnn_stats": dmpnn_stats,
    }
    target_pt = os.path.join(output_dir, "ppbr_tri_hybrid_sota.pt")
    torch.save(package, target_pt)

    # Standard model.pt copy for generic loader fallback
    target_standard = os.path.join(output_dir, "model.pt")
    torch.save(package, target_standard)

    config = {
        "type": "tri_hybrid_stacker",
        "task_type": "regression",
        "primary_task": "ppbr_az",
        "metric_name": "r2",
        "blending_weights": {
            "w_gnn": float(blending_weights[0]),
            "w_gbdt": float(blending_weights[1]),
            "w_chemberta": float(blending_weights[2]),
        },
        "calibration": {
            "alpha": float(calibration_params[0]),
            "beta": float(calibration_params[1]),
        },
    }
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    if benchmark_meta:
        with open(os.path.join(output_dir, "training_meta.json"), "w", encoding="utf-8") as f:
            json.dump(benchmark_meta, f, indent=2)

    manifest = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "model_type": "tri_hybrid_stacker",
        "task_type": "regression",
        "primary_task": "ppbr_az",
        "weights_file": "ppbr_tri_hybrid_sota.pt",
        "config_file": "config.json",
        "benchmark_meta": benchmark_meta or {},
    }
    with open(os.path.join(output_dir, "export_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def export_dti_package(
    model: torch.nn.Module,
    model_config: Dict[str, Any],
    output_dir: str,
    scaler_meta: Optional[Dict[str, Any]] = None,
    checkpoint_name: str = "model.pt",
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Package and export DTI model, architecture config, and normalization scaler for serving.

    Args:
        model          : Trained GraphDTAModel instance.
        model_config   : Model configuration dict.
        output_dir     : Directory to store exported files.
        scaler_meta    : Optional normalization stats (y_mean, y_std, log_transform).
        checkpoint_name: Name of weights file (default "model.pt").
        extra_meta     : Optional evaluation or training metrics.

    Returns:
        Manifest dict.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Save weights
    weights_path = os.path.join(output_dir, checkpoint_name)
    torch.save(model.state_dict(), weights_path)

    # 2. Save model config (ensure type="graph_dta" and is_dta=True)
    config_copy = dict(model_config)
    if "type" not in config_copy:
        config_copy["type"] = "graph_dta"
    config_copy["is_dta"] = True

    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_copy, f, indent=2)

    # 3. Save scaler metadata
    if scaler_meta:
        scaler_path = os.path.join(output_dir, "scaler.json")
        with open(scaler_path, "w", encoding="utf-8") as f:
            json.dump(scaler_meta, f, indent=2)

    # 4. Save optional training metadata
    if extra_meta:
        meta_path = os.path.join(output_dir, "training_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(extra_meta, f, indent=2)

    # 5. Generate manifest
    param_count = sum(p.numel() for p in model.parameters())
    manifest = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "model_type": config_copy["type"],
        "task_type": "dta",
        "parameter_count": param_count,
        "weights_file": checkpoint_name,
        "config_file": "config.json",
        "has_scaler": scaler_meta is not None,
        "extra_meta": extra_meta or {},
    }
    manifest_path = os.path.join(output_dir, "export_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def export_torchscript_model(
    model: torch.nn.Module,
    example_inputs: Any,
    output_dir: str,
    file_name: str = "model.torchscript.pt",
    method: str = "trace",
    strict: bool = False,
) -> str:
    """Compile and export PyTorch model into TorchScript format for C++ serving and GIL-free speedup.

    Args:
        model: Model module in eval mode.
        example_inputs: Sample input tensor or tuple for tracing.
        output_dir: Destination directory.
        file_name: Output filename (default 'model.torchscript.pt').
        method: 'trace' (torch.jit.trace) or 'script' (torch.jit.script).
        strict: Strict tracing validation.

    Returns:
        Absolute path to the saved TorchScript artifact.
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    if method == "script":
        scripted = torch.jit.script(model)
    else:
        if isinstance(example_inputs, (list, tuple)):
            scripted = torch.jit.trace(model, example_inputs, strict=strict)
        else:
            scripted = torch.jit.trace(model, (example_inputs,), strict=strict)

    out_path = os.path.join(output_dir, file_name)
    scripted.save(out_path)
    return out_path


def export_onnx_model(
    model: torch.nn.Module,
    example_inputs: Any,
    output_dir: str,
    file_name: str = "model.onnx",
    input_names: Optional[List[str]] = None,
    output_names: Optional[List[str]] = None,
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    opset_version: int = 17,
) -> str:
    """Export PyTorch neural network to Open Neural Network Exchange (ONNX) format.

    Args:
        model: Model module in eval mode.
        example_inputs: Sample inputs matching forward signature.
        output_dir: Destination directory.
        file_name: Output filename (default 'model.onnx').
        input_names: Input tensor name identifiers.
        output_names: Output tensor name identifiers.
        dynamic_axes: Dict mapping tensor names to dynamic dimension mappings (e.g. batch size).
        opset_version: ONNX operator set version (default 17).

    Returns:
        Absolute path to the saved ONNX artifact.
    """
    import onnx

    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    out_path = os.path.join(output_dir, file_name)
    input_names = input_names or ["input"]
    output_names = output_names or ["output"]

    if dynamic_axes is None:
        dynamic_axes = {
            input_names[0]: {0: "batch_size"},
            output_names[0]: {0: "batch_size"},
        }

    if not isinstance(example_inputs, (tuple, list)):
        args = (example_inputs,)
    else:
        args = example_inputs

    torch.onnx.export(
        model,
        args,
        out_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    onnx_model = onnx.load(out_path)
    onnx.checker.check_model(onnx_model)

    return out_path


def load_onnx_inference_session(onnx_path: str) -> Any:
    """Load ONNX model into high-throughput ONNX Runtime InferenceSession.

    Args:
        onnx_path: Path to .onnx file.

    Returns:
        onnxruntime.InferenceSession instance.
    """
    import onnxruntime as ort

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = min(4, os.cpu_count() or 1)

    session = ort.InferenceSession(onnx_path, sess_options, providers=["CPUExecutionProvider"])
    return session


class OptimizedServingRuntime:
    """Unified high-performance runtime engine supporting PyTorch, TorchScript, and ONNX backends."""

    def __init__(
        self,
        pytorch_model: Optional[torch.nn.Module] = None,
        torchscript_path: Optional[str] = None,
        onnx_path: Optional[str] = None,
    ):
        self.pytorch_model = pytorch_model
        self.torchscript_model = None
        self.onnx_session = None
        self.active_backend = "pytorch"

        if onnx_path and os.path.exists(onnx_path):
            try:
                self.onnx_session = load_onnx_inference_session(onnx_path)
                self.active_backend = "onnx"
            except Exception:
                pass

        if (
            self.active_backend == "pytorch"
            and torchscript_path
            and os.path.exists(torchscript_path)
        ):
            try:
                self.torchscript_model = torch.jit.load(torchscript_path, map_location="cpu")
                self.torchscript_model.eval()
                self.active_backend = "torchscript"
            except Exception:
                pass

    def run_inference(self, inputs: Any) -> Any:
        """Execute accelerated inference via the fastest available backend."""
        import numpy as np

        if self.active_backend == "onnx" and self.onnx_session is not None:
            if isinstance(inputs, torch.Tensor):
                np_in = inputs.detach().cpu().numpy()
            else:
                np_in = np.asarray(inputs)

            input_name = self.onnx_session.get_inputs()[0].name
            outputs = self.onnx_session.run(None, {input_name: np_in})
            return outputs[0]

        elif self.active_backend == "torchscript" and self.torchscript_model is not None:
            if isinstance(inputs, np.ndarray):
                t_in = torch.from_numpy(inputs)
            else:
                t_in = inputs
            with torch.no_grad():
                out = self.torchscript_model(t_in)
                if isinstance(out, torch.Tensor):
                    return out.detach().cpu().numpy()
                return np.asarray(out)

        elif self.pytorch_model is not None:
            if isinstance(inputs, np.ndarray):
                t_in = torch.from_numpy(inputs)
            else:
                t_in = inputs
            with torch.no_grad():
                out = self.pytorch_model(t_in)
                if isinstance(out, torch.Tensor):
                    return out.detach().cpu().numpy()
                return np.asarray(out)

        raise RuntimeError("No active model or backend initialized in OptimizedServingRuntime.")
