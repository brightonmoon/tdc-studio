"""Tests for TorchScript and ONNX serving optimization and export."""

import os
import numpy as np
import pytest
import torch
import torch.nn as nn

from tdc_studio.serving.exporter import (
    OptimizedServingRuntime,
    export_onnx_model,
    export_torchscript_model,
    load_onnx_inference_session,
)


class SimpleADMETMLP(nn.Module):
    """Simple 2-layer MLP model representing an ADMET descriptor predictor."""

    def __init__(self, in_features: int = 16, hidden: int = 32, out_features: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@pytest.fixture
def mock_mlp_and_inputs():
    torch.manual_seed(42)
    model = SimpleADMETMLP(in_features=16, hidden=32, out_features=1)
    model.eval()
    example_input = torch.randn(4, 16)
    return model, example_input


def test_export_torchscript_model(tmp_path, mock_mlp_and_inputs):
    """Verify TorchScript compilation, export, and load equivalence."""
    model, example_input = mock_mlp_and_inputs
    out_dir = str(tmp_path / "ts_export")

    ts_path = export_torchscript_model(
        model=model,
        example_inputs=example_input,
        output_dir=out_dir,
        file_name="admet_model.torchscript.pt",
        method="trace",
    )

    assert os.path.exists(ts_path)

    # Load and compare numerical outputs
    loaded_ts = torch.jit.load(ts_path)
    loaded_ts.eval()

    with torch.no_grad():
        orig_out = model(example_input)
        ts_out = loaded_ts(example_input)

    assert torch.allclose(orig_out, ts_out, atol=1e-5)


def test_export_onnx_model_and_inference_session(tmp_path, mock_mlp_and_inputs):
    """Verify ONNX export, checker validation, and ONNX Runtime execution."""
    model, example_input = mock_mlp_and_inputs
    out_dir = str(tmp_path / "onnx_export")

    onnx_path = export_onnx_model(
        model=model,
        example_inputs=example_input,
        output_dir=out_dir,
        file_name="admet_model.onnx",
        input_names=["molecular_descriptors"],
        output_names=["admet_prediction"],
    )

    assert os.path.exists(onnx_path)

    # Run inference with onnxruntime
    session = load_onnx_inference_session(onnx_path)
    input_name = session.get_inputs()[0].name
    assert input_name == "molecular_descriptors"

    np_input = example_input.detach().cpu().numpy()
    ort_outputs = session.run(None, {input_name: np_input})
    ort_pred = ort_outputs[0]

    with torch.no_grad():
        py_pred = model(example_input).detach().cpu().numpy()

    assert np.allclose(py_pred, ort_pred, atol=1e-5)


def test_optimized_serving_runtime(tmp_path, mock_mlp_and_inputs):
    """Verify OptimizedServingRuntime backend prioritization (ONNX -> TorchScript -> PyTorch)."""
    model, example_input = mock_mlp_and_inputs
    out_dir = str(tmp_path / "unified_runtime")

    ts_path = export_torchscript_model(model, example_input, out_dir, "model.torchscript.pt")
    onnx_path = export_onnx_model(model, example_input, out_dir, "model.onnx")

    # 1. Initialize with ONNX preferred
    runtime_onnx = OptimizedServingRuntime(
        pytorch_model=model,
        torchscript_path=ts_path,
        onnx_path=onnx_path,
    )
    assert runtime_onnx.active_backend == "onnx"
    out_onnx = runtime_onnx.run_inference(example_input)
    assert out_onnx.shape == (4, 1)

    # 2. Initialize with TorchScript (no ONNX)
    runtime_ts = OptimizedServingRuntime(
        pytorch_model=model,
        torchscript_path=ts_path,
        onnx_path="non_existent.onnx",
    )
    assert runtime_ts.active_backend == "torchscript"
    out_ts = runtime_ts.run_inference(example_input)
    assert np.allclose(out_onnx, out_ts, atol=1e-5)

    # 3. Fallback to native PyTorch
    runtime_py = OptimizedServingRuntime(
        pytorch_model=model,
        torchscript_path="non_existent.pt",
        onnx_path="non_existent.onnx",
    )
    assert runtime_py.active_backend == "pytorch"
    out_py = runtime_py.run_inference(example_input)
    assert np.allclose(out_onnx, out_py, atol=1e-5)
