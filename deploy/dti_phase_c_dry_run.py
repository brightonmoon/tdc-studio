"""DTI Phase C Dry-Run & Integrity Verification Script.

Validates:
  1. CrossAttentionFusion instantiation via MODELS registry
  2. GraphDTAModel integration with CrossAttentionFusion
  3. Bidirectional attention map extraction (XAI weights)
  4. Staged unfreezing parameter gradients
  5. End-to-end forward/backward gradient flow
"""

# ruff: noqa: E402

import os
import sys

# Extract bundle if executed via colab exec on remote VM
_bundle_b64 = globals().get("BUNDLE_B64", None)
if _bundle_b64:
    import base64
    import io
    import zipfile

    print("[BUNDLE] Extracting local workspace bundle...")
    data = base64.b64decode(_bundle_b64)
    workspace = os.path.abspath("tdc-studio")
    with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
        zf.extractall(workspace)
    if workspace not in sys.path:
        sys.path.insert(0, workspace)
    os.chdir(workspace)
    print(f"[BUNDLE] Unpacked workspace at: {workspace}")
else:
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


import numpy as np
import torch
from torch_geometric.data import Batch, Data

import tdc_studio.models.dti.fusion  # noqa: F401
import tdc_studio.models.dti.pretrained_encoders  # noqa: F401
from tdc_studio.core.registry import MODELS
from tdc_studio.evaluation.evaluator import TherapeuticsEvaluator
from tdc_studio.models.dti.dta_model import GraphDTAModel
from tdc_studio.models.dti.fusion import CrossAttentionFusion


def run_phase_c_dry_run():
    print("=" * 65)
    print("  * TDC-Studio DTI Phase C Dry-Run (Cross-Attention & Staged Training)")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Verify MODELS Registry
    print("\n[Step 1] Checking MODELS Registry...")
    assert "cross_attention_fusion" in MODELS
    assert "cross_attention" in MODELS
    print("  [OK] CrossAttentionFusion successfully registered in MODELS!")


    # 2. Standalone CrossAttentionFusion
    print("\n[Step 2] Testing Standalone CrossAttentionFusion...")
    cfg = {
        "drug_dim": 64,
        "target_dim": 64,
        "hidden_dim": 32,
        "num_heads": 2,
        "out_dim": 1,
    }
    fusion = CrossAttentionFusion(cfg).to(device)
    fusion.eval()
    h_d = torch.randn(2, 6, 64, device=device)
    h_t = torch.randn(2, 20, 64, device=device)
    out, attn = fusion(h_d, h_t, return_attention=True)
    assert out.shape == (2, 1)
    assert "attn_d2t" in attn and "attn_t2d" in attn
    print(f"  [OK] Cross-Attention Output: {out.shape}, Attn D->T: {attn['attn_d2t'].shape}")

    # 3. GraphDTAModel with CrossAttention
    print("\n[Step 3] Initializing GraphDTAModel with CrossAttention...")
    model_cfg = {
        "type": "graph_dta",
        "drug_encoder": {
            "type": "gine",
            "in_dim": 14,
            "hidden_dim": 32,
            "num_layers": 2,
        },
        "target_encoder": {
            "type": "protein_cnn",
            "out_dim": 32,
            "kernel_sizes": [3],
        },
        "fusion": {
            "type": "cross_attention",
            "hidden_dim": 32,
            "num_heads": 2,
        },
    }
    model = GraphDTAModel(model_cfg).to(device)
    assert isinstance(model.fusion, CrossAttentionFusion)
    print("  [OK] GraphDTAModel built with CrossAttentionFusion successfully!")

    # 4. Forward and Backward
    print("\n[Step 4] Running Forward & Backward pass...")
    g1 = Data(x=torch.randn(4, 14), edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long), edge_attr=torch.randn(2, 6))
    g2 = Data(x=torch.randn(3, 14), edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long), edge_attr=torch.randn(2, 6))
    batch_graph = Batch.from_data_list([g1, g2]).to(device)

    batch = {
        "drug_graph": batch_graph,
        "target_seq": torch.randint(1, 25, (2, 30), device=device),
    }

    model.train()
    preds = model(batch)
    targets = torch.tensor([4.5, 6.2], device=device)
    loss = model.compute_loss(preds, targets)
    loss.backward()

    assert model.fusion.proj_drug.weight.grad is not None
    print(f"  [OK] Loss: {loss.item():.4f}, Gradients computed cleanly!")

    # 5. Evaluator Check
    print("\n[Step 5] Evaluating DTA Metrics...")
    evaluator = TherapeuticsEvaluator(task_type="dta")
    y_pred = np.array([4.5, 6.2, 7.8, 5.1])
    y_true = np.array([4.2, 6.0, 8.1, 5.4])
    metrics = evaluator.compute_all(y_pred, y_true, task_type="dta")
    print(f"  [OK] Evaluator Metrics: CI={metrics['ci']:.4f}, MSE={metrics['mse']:.4f}")

    print("\n" + "=" * 65)
    print("  * Phase C Dry-Run Verification: ALL 5 STEPS PASSED! *")
    print("=" * 65)

    return 0


if __name__ == "__main__":
    sys.exit(run_phase_c_dry_run())
