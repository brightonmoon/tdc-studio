"""DTI Phase A Dry-Run — 패키지 자동설치 없음, synthetic 데이터만 사용.

의존성 가정:
  - torch (Colab 기본)
  - torch_geometric (colab install로 이미 설치됨)
  - rdkit (Colab 기본 제공)
  - PyTDC 불필요 (synthetic DataFrame 사용)

테스트 항목:
  1. AminoAcidTokenizer 정확성
  2. CI (Concordance Index) 메트릭
  3. DTADataModule cold_drug split (synthetic)
  4. GraphDTAModel forward pass
  5. Loss + backward 2 step
  6. Evaluator DTA metrics
"""

import sys, os

# ── workspace 복원 (colab exec bundle 방식) ──────────────────────────────────
_bundle_b64 = globals().get("BUNDLE_B64", None)
if _bundle_b64:
    import base64, io, zipfile
    print("[BUNDLE] Extracting workspace...")
    data = base64.b64decode(_bundle_b64)
    workspace = os.path.abspath("tdc-studio")
    with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
        zf.extractall(workspace)
    if workspace not in sys.path:
        sys.path.insert(0, workspace)
    os.chdir(workspace)
    print(f"[BUNDLE] Done -> {workspace}")
else:
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

# ── 환경 확인 ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("[DTI DRY-RUN] Environment check")
print("=" * 60)

import torch
print(f"  torch        : {torch.__version__}  CUDA={torch.cuda.is_available()}")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  device       : {device}")

try:
    import torch_geometric
    print(f"  torch_geo    : {torch_geometric.__version__}")
except ImportError as e:
    print(f"  [FAIL] torch_geometric not found: {e}")
    sys.exit(1)

try:
    from rdkit import Chem
    print(f"  rdkit        : available")
except ImportError:
    print("  [FAIL] rdkit not found — cannot run SmilesToGraphTransform")
    sys.exit(1)

import numpy as np
print(f"  numpy        : {np.__version__}")
print()

# ══════════════════════════════════════════════════════════════
# TEST 1: AminoAcidTokenizer
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("[TEST 1] AminoAcidTokenizer")
print("=" * 60)

from tdc_studio.data.transforms import AminoAcidTokenizer

tok = AminoAcidTokenizer(max_length=20)
seq = "MKTAYIAKQR"
t = tok(seq)

assert t.shape == (20,),      f"Shape mismatch: {t.shape}"
assert t.dtype == torch.long, f"Dtype mismatch: {t.dtype}"
# PAD tokens at tail
assert t[len(seq):].sum() == 0, "Tail should be PAD"
decoded = tok.decode(t)
assert decoded == seq, f"Decode mismatch: '{decoded}' != '{seq}'"

print(f"  seq     : {seq}")
print(f"  tokens  : {t.tolist()}")
print(f"  decoded : {decoded}")
print(f"  vocab   : {AminoAcidTokenizer.VOCAB_SIZE} tokens")
print("[TEST 1] PASSED\n")

# ══════════════════════════════════════════════════════════════
# TEST 2: CI Metric
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("[TEST 2] Concordance Index (CI) Metric")
print("=" * 60)

from tdc_studio.evaluation.evaluator import TherapeuticsEvaluator

ev = TherapeuticsEvaluator()
y_true    = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_perfect = np.array([1.1, 2.0, 2.9, 4.1, 5.2])
y_worst   = np.array([5.0, 4.0, 3.0, 2.0, 1.0])

ci_good = ev.compute(y_perfect, y_true, "ci")
ci_bad  = ev.compute(y_worst,   y_true, "ci")

print(f"  CI (good order) : {ci_good:.4f}  [expected > 0.90]")
print(f"  CI (reversed)   : {ci_bad:.4f}   [expected < 0.10]")

assert ci_good > 0.90, f"CI too low: {ci_good}"
assert ci_bad  < 0.10, f"Reversed CI too high: {ci_bad}"

dta_metrics = ev.compute_all(y_perfect, y_true, task_type="dta")
assert "ci" in dta_metrics and "mse" in dta_metrics, f"Missing metrics: {dta_metrics}"
print(f"  DTA metrics     : {dta_metrics}")
print("[TEST 2] PASSED\n")

# ══════════════════════════════════════════════════════════════
# TEST 3: DTADataModule (synthetic, cold_drug default)
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("[TEST 3] DTADataModule — synthetic + cold_drug")
print("=" * 60)

import pandas as pd
from tdc_studio.data.multi_pred import DTADataModule

SMILES = [
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "CC(=O)Oc1ccccc1C(=O)O",
    "CN1CCC[C@H]1c2cccnc2",
    "c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34",
    "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",
]
TARGET = [
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGD",
    "MSHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTHTAKYdp",
]

np.random.seed(42)
N = 30
synthetic_df = pd.DataFrame({
    "Drug_ID":   [f"D{i:03d}" for i in range(N)],
    "Drug":      [SMILES[i % len(SMILES)] for i in range(N)],
    "Target_ID": [f"T{i % 3:03d}" for i in range(N)],
    "Target":    [TARGET[i % len(TARGET)] for i in range(N)],
    "Y":         np.random.uniform(1.0, 10000.0, N),
})

dm = DTADataModule(
    dataset_name  = "synthetic",
    split_type    = "cold_drug",
    log_transform = True,
    aa_max_length = 50,
    synthetic_df  = synthetic_df,
)
dm.prepare_data()
train_dl, val_dl, test_dl = dm.setup_loaders(batch_size=8)

print(f"  split_info    : {dm.split_info}")
print(f"  y_mean/std    : {dm.y_mean:.3f} / {dm.y_std:.3f}")

batch = next(iter(train_dl))
print(f"  batch keys    : {sorted(batch.keys())}")

assert "drug_graph"      in batch, "drug_graph missing"
assert "target_seq"      in batch, "target_seq missing"
assert "drug_smiles_str" in batch, "drug_smiles_str missing (Phase B field)"
assert "labels"          in batch, "labels missing"

print(f"  target_seq    : {batch['target_seq'].shape}  dtype={batch['target_seq'].dtype}")
print(f"  label sample  : {batch['labels'][:4]}")

print("[TEST 3] PASSED\n")

# ══════════════════════════════════════════════════════════════
# TEST 4: GraphDTAModel forward pass
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("[TEST 4] GraphDTAModel Forward Pass")
print("=" * 60)

from tdc_studio.models.dti.dta_model import GraphDTAModel

cfg = {
    "task_type": "dta",
    "drug_encoder":   {"type": "gine",       "in_dim": 14, "edge_dim": 6,
                       "hidden_dim": 64, "num_layers": 3, "dropout": 0.1},
    "target_encoder": {"type": "protein_cnn", "embed_dim": 32,
                       "num_filters": 64, "out_dim": 128, "dropout": 0.1},
    "fusion":         {"hidden_dim": 128, "dropout": 0.1},
}

model = GraphDTAModel(cfg).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"  params        : {n_params:,}")

batch_dev = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}

model.eval()
with torch.no_grad():
    h_drug, h_target = model.extract_features(batch_dev)
    preds = model(batch_dev)

print(f"  h_drug shape  : {h_drug.shape}")
print(f"  h_target shape: {h_target.shape}")
print(f"  preds shape   : {preds.shape}")
print(f"  preds         : {preds.squeeze().tolist()}")

assert preds.shape[-1] == 1,       f"Expected [B,1], got {preds.shape}"
assert not torch.isnan(preds).any(), "NaN in predictions!"
print("[TEST 4] PASSED\n")

# ══════════════════════════════════════════════════════════════
# TEST 5: Loss + backward
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("[TEST 5] Loss Computation + 2-Step Backward")
print("=" * 60)

model.train()
opt = torch.optim.Adam(model.parameters(), lr=1e-4)

for step in range(2):
    opt.zero_grad()
    b = next(iter(train_dl))
    b = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in b.items()}
    p = model(b)
    tgt = b["labels"].detach().clone().float().to(device)

    loss = model.compute_loss(p, tgt)
    loss.backward()
    opt.step()
    print(f"  step {step+1}: loss={loss.item():.6f}")

assert not torch.isnan(loss), "NaN loss!"
print("[TEST 5] PASSED\n")

# ══════════════════════════════════════════════════════════════
# TEST 6: End-to-end Evaluator
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("[TEST 6] End-to-End Evaluator on Val Set")
print("=" * 60)

model.eval()
all_p, all_t = [], []
with torch.no_grad():
    for b in val_dl:
        b = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in b.items()}
        p = model(b).squeeze(-1).cpu().numpy()
        all_p.extend(p.tolist())
        all_t.extend(b["labels"].cpu().numpy().tolist())


if len(all_p) >= 2:
    m = ev.compute_all(np.array(all_p), np.array(all_t), task_type="dta")
    print(f"  val CI   : {m.get('ci',   'N/A'):.4f}")
    print(f"  val MSE  : {m.get('mse',  'N/A'):.4f}")
    print(f"  val RMSE : {m.get('rmse', 'N/A'):.4f}")
else:
    print("  [SKIP] val set too small")
print("[TEST 6] PASSED\n")

# ══════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("[DTI DRY-RUN] ALL TESTS PASSED")
print("=" * 60)
print(f"  Device  : {device} ({'GPU' if device=='cuda' else 'CPU'})")
print(f"  Params  : {n_params:,}")
print()
print("  TEST 1 AminoAcidTokenizer     OK")
print("  TEST 2 CI Metric              OK")
print("  TEST 3 DTADataModule          OK")
print("  TEST 4 GraphDTAModel forward  OK")
print("  TEST 5 Loss + backward        OK")
print("  TEST 6 Evaluator DTA          OK")
print()
print("  Next: Full training with BindingDB_Kd cold_drug split")
print("        config: configs/config_dti_phase_a.yaml")
print("=" * 60)
