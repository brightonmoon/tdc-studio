# TDC-Studio

Modular MLOps Studio for Therapeutics Data Commons (TDC) with Cloud GPU Orchestration.

---

## Key Features

1. **Modular Architecture & Loose Coupling**:
   - **Registry Pattern**: Dynamic plugin discovery for models, datasets, and featurizers via `@MODELS.register` and `@DATASETS.register`.
   - **TDC Data Modules**: Clean abstraction over TDC single-prediction (ADMET) and multi-prediction (DTA/BindingDB) datasets with custom PyTorch Geometric graph collators.
   - **Dynamic Loss Strategies**: Automatic loss configuration (MSE for regression, BCEWithLogits for classification).
   - **Optuna HPO & W&B**: Declarative YAML search space parsing, data re-use, and early-stopping (pruning).
   - **FastAPI Non-Blocking Serving**: End-to-end `InferencePipeline` offloading CPU-intensive RDKit featurization and PyTorch inference to threadpools with `/healthz` readiness probes.

2. **Cloud GPU Orchestration (`google-colab-cli`)**:
   - Eliminate heavy local GPU burdens by dispatching jobs to ephemeral Google Colab runtimes (T4, L4, A100).
   - One-click standalone notebook generation (`tdc-studio remote export-notebook`).

3. **Ultra-Fast Development with UV**:
   - Pinned to **Python 3.11** for guaranteed binary wheel compatibility with `PyTDC`, `rdkit`, `torch-geometric`, and `onnxruntime`.

---

## Quickstart

### 1. Installation

```bash
# Clone and sync dependencies using uv
git clone <repo-url>
cd tdc-studio
uv sync --extra dev
```

### 2. Linting and Testing (Zero Heavy Training)

```bash
# Static analysis and linting
uv run ruff check tdc_studio tests

# Run unit tests
uv run pytest -v
```

### 3. CLI Commands

```bash
# Local baseline training (supports --dry-run)
uv run tdc-studio train --config configs/config.yaml --dry-run

# Hyperparameter optimization with Optuna
uv run tdc-studio tune --config configs/config.yaml --n-trials 5 --dry-run

# Cloud GPU Remote Training (Google Colab CLI)
uv run tdc-studio remote run --gpu a100 --command "tdc-studio tune --config configs/config.yaml" --dry-run

# Export Colab Notebook for interactive browser execution
uv run tdc-studio remote export-notebook --output tdc_colab.ipynb

# Start Serving API
uv run tdc-studio serve --port 8000
```

### 4. Production Container Deployment

```bash
docker compose -f deploy/docker-compose.yml up -d
```
