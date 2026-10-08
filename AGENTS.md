# AGENTS.md - Agent Operating Guidelines for TDC-Studio

This file sets foundational rules and behavioral constraints for all AI coding agents (Antigravity, Cursor, Claude Code, Devin) operating within the `tdc-studio` repository.

---

## 🚨 CRITICAL RULE: Machine Learning & Benchmark Execution Constraints

### 1. Absolute Prohibition of Heavy Local Training (로컬 무거운 학습 금지)
- **NEVER** run full training loops, hyperparameter optimization, or heavy benchmark scripts directly on the local Windows workstation (e.g. `uv run python train.py`, `python scripts/benchmark_*.py`).
- **Local execution is strictly restricted to 1-step dry-runs** using `--dry-run --local` to verify tensor shapes and config syntax.
- All actual model training MUST be executed on **Google Colab Cloud GPU** (`colab exec` or `colab run`).

### 2. Mandatory Skill Usage: `tdc-colab-wandb-training`
Whenever the user requests model training, fine-tuning, or benchmark execution:
- **Activate and follow**: [`.agents/skills/tdc-colab-wandb-training/SKILL.md`](file:///C:/Users/xps/orca/workspaces/tdc-studio/.agents/skills/tdc-colab-wandb-training/SKILL.md)
- **Use the standardized pipeline runner**:
  ```bash
  uv run python scripts/run_training_pipeline.py --config <CONFIG_PATH>
  ```
- **Do NOT write ad-hoc training scripts** or modify dataset loaders arbitrarily. Rely strictly on declarative YAML configs under `configs/`.

### 3. Mandatory Experiment Tracking & Observability (W&B 모니터링 의무)
- Every training execution must be logged to **Weights & Biases (W&B)**.
- Agents must immediately locate the W&B Run URL in stdout (e.g. `https://wandb.ai/tdc-studio/...`) and report it to the user.
- Agents must not conclude the task until:
  1. The remote training job completes.
  2. Model checkpoints and manifests are synchronized to `models/export/` via `scripts/sync_wandb_models.py`.
  3. A concise, structured Markdown report of validation metrics is delivered.

### 4. Remote Environment & Account Management
- Use `uv run tdc-studio remote status` or `.\scripts\colab_switch.ps1 list` to check active sessions.
- In case of GPU quota errors (`TooManyAssignmentsError`), rotate to an available backup account using `.\scripts\colab_switch.ps1 use <account>` or specify `--auto-switch`.
