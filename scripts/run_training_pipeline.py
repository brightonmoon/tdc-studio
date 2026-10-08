"""Standardized End-to-End Cloud Training Pipeline Runner for TDC-Studio.

Orchestrates the 5-phase deterministic training lifecycle:
  Phase 1: Pre-flight Verification (Colab CLI, W&B API Key, Config validation)
  Phase 2: Local 1-Step Dry-Run Validation (Syntax & data schema sanity check)
  Phase 3: Remote Colab Cloud GPU Execution (Code bundling, argument injection, remote execution)
  Phase 4: Real-Time Monitoring & Observability (Log streaming, W&B tracking)
  Phase 5: Post-Training Artifact Synchronization (Model checkpoint & manifest sync via W&B)
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


def print_banner(text: str) -> None:
    sep = "=" * 70
    print(f"\n{sep}\n  {text}\n{sep}", flush=True)


def run_cmd(cmd: list[str], check: bool = True, env: Optional[dict] = None) -> subprocess.CompletedProcess:
    """Run command with live streaming output."""
    print(f"[Exec] {' '.join(cmd)}", flush=True)
    custom_env = os.environ.copy()
    custom_env["PYTHONIOENCODING"] = "utf-8"
    custom_env["PYTHONUTF8"] = "1"
    if env:
        custom_env.update(env)

    res = subprocess.run(cmd, env=custom_env)
    if check and res.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {res.returncode}: {' '.join(cmd)}")
    return res


def check_wandb_status() -> tuple[bool, Optional[str]]:
    """Verify W&B API Key availability."""
    try:
        import wandb

        api = wandb.Api()
        key = api.api_key
        entity = api.default_entity
        return bool(key), entity
    except Exception:
        key = os.environ.get("WANDB_API_KEY")
        return bool(key), None


def get_active_colab_session() -> Optional[str]:
    """Retrieve active Colab session identifier from colab-cli sessions.json or cli."""
    import json

    # 1. Check local config sessions.json directly for registered session keys
    home = Path.home()
    config_paths = [
        home / ".config" / "colab-cli" / "sessions.json",
        Path(r"C:\Users\xps\.colab_munhyeongdo4\.config\colab-cli\sessions.json"),
    ]
    for cp in config_paths:
        if cp.exists():
            try:
                data = json.loads(cp.read_text(encoding="utf-8"))
                for s_name in data.keys():
                    return s_name
            except Exception:
                pass

    if not shutil.which("colab"):
        return None
    try:
        res = subprocess.run(["colab", "sessions"], capture_output=True, text=True, check=False)
        lines = res.stdout.splitlines()
        for line in lines:
            line = line.strip()
            # Match session names like 'gpu-t4-...' or lines containing online/running
            if any(status in line.upper() for status in ["RUNNING", "ACTIVE", "ASSIGNED", "ONLINE", "T4", "GPU"]):
                parts = line.split()
                # Check for explicit gpu- prefixed ID first
                for p in parts:
                    clean = p.strip("*-[]|,")
                    if clean.startswith("gpu-"):
                        return clean
                for p in parts:
                    clean = p.strip("*-[]|,")
                    if "session" in clean or len(clean) > 8:
                        return clean
        # Fallback: check if any non-header bullet exists
        for line in lines:
            if line.startswith("*") or line.startswith("-"):
                parts = line.lstrip("*- ").split()
                if parts:
                    return parts[0]
    except Exception:
        pass
    return None


def execute_pipeline(
    config_path: str,
    session: Optional[str] = None,
    gpu: str = "T4",
    skip_dry_run: bool = False,
    no_sync: bool = False,
    epochs: Optional[int] = None,
) -> int:
    start_time = time.time()
    resolved_config = Path(config_path).resolve()
    if not resolved_config.exists():
        print(f"[Error] Config file not found: {resolved_config}", file=sys.stderr)
        return 1

    print_banner(f"TDC-Studio Standardized Training Pipeline: {resolved_config.name}")

    # =========================================================================
    # Phase 1: Pre-Flight Verification
    # =========================================================================
    print("\n[Phase 1/5] Pre-Flight Verification")
    wandb_ok, wandb_entity = check_wandb_status()
    if not wandb_ok:
        print("[Warning] W&B API key not detected in environment or netrc!")
        print("          Runs may execute offline without cloud dashboard tracking.")
        print("          Set WANDB_API_KEY or run `uv run wandb login` to enable full tracking.")
    else:
        print(f"  [OK] W&B Authentication verified (Entity/Default: {wandb_entity or 'tdc-studio'})")

    if not shutil.which("colab"):
        print("[Error] Google Colab CLI ('colab') is not found on PATH.", file=sys.stderr)
        print("        Install it via `uv tool install google-colab-cli` or `pip install google-colab-cli`.")
        return 1
    print("  [OK] Google Colab CLI found on PATH.")

    # Determine Colab Session
    target_session = session or get_active_colab_session()
    if target_session:
        print(f"  [OK] Target Colab GPU Session: {target_session}")
    else:
        print(f"  [Notice] No active persistent session found. Will dispatch ephemeral GPU job ({gpu}).")

    # =========================================================================
    # Phase 2: Local 1-Step Dry-Run Validation
    # =========================================================================
    print("\n[Phase 2/5] Local 1-Step Dry-Run Validation (Zero Heavy Local Compute)")
    if skip_dry_run:
        print("  [Skip] Local dry-run skipped by user request.")
    else:
        print("  Validating pipeline schema with 1-step dry run on local CPU...")
        dry_run_cmd = [
            "uv", "run", "tdc-studio", "train",
            "--config", str(resolved_config),
            "--dry-run",
            "--local",
        ]
        res = run_cmd(dry_run_cmd, check=False)
        if res.returncode != 0:
            print(f"\n[Validation Failed] Dry-run failed with code {res.returncode}!", file=sys.stderr)
            print("Aborting before dispatching to remote GPU to conserve compute units.", file=sys.stderr)
            return res.returncode
        print("  [OK] Dry-run passed successfully. Config and model architecture are valid.")

    # =========================================================================
    # Phase 3 & 4: Remote Colab Cloud GPU Execution & Monitoring
    # =========================================================================
    print("\n[Phase 3 & 4/5] Remote Cloud GPU Execution & Real-Time Monitoring")
    try:
        rel_config = str(resolved_config.relative_to(Path.cwd())).replace("\\", "/")
    except ValueError:
        rel_config = str(resolved_config).replace("\\", "/")

    train_args = ["train", "--local", "--config", rel_config]
    if epochs:
        train_args.extend(["--epochs", str(epochs)])

    remote_train_cmd = f"tdc-studio {' '.join(train_args)}"

    if target_session:
        print(f"  Dispatching task to active Colab session '{target_session}' via PowerShell helper...")
        ps_script = Path("scripts/colab_exec.ps1").resolve()
        runner_job = Path("deploy/colab_runner_job.py").resolve()

        if ps_script.exists():
            exec_cmd = [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy", "Bypass",
                "-File", str(ps_script),
                target_session,
                str(runner_job),
                remote_train_cmd,
            ]
        else:
            exec_cmd = [
                "uv", "run", "tdc-studio", "remote", "exec",
                "-s", target_session,
                "-f", str(runner_job),
                "--arg", remote_train_cmd,
            ]
    else:
        print(f"  Dispatching ephemeral cloud GPU job ({gpu}) via `tdc-studio remote run`...")
        exec_cmd = [
            "uv", "run", "tdc-studio", "remote", "run",
            "--gpu", gpu,
            "--command", remote_train_cmd,
        ]

    exec_res = run_cmd(exec_cmd, check=False)
    if exec_res.returncode != 0:
        print(f"\n[Execution Failed] Remote Colab job exited with code {exec_res.returncode}!", file=sys.stderr)
        return exec_res.returncode
    print("  [OK] Remote Colab training execution finished successfully.")

    # =========================================================================
    # Phase 5: Post-Training Artifact Synchronization
    # =========================================================================
    print("\n[Phase 5/5] Post-Training Artifact Synchronization & Manifest Export")
    if no_sync:
        print("  [Skip] Artifact sync skipped by user request.")
    else:
        sync_script = Path("scripts/sync_wandb_models.py").resolve()
        if sync_script.exists():
            print("  Synchronizing latest model artifacts and benchmark metrics from W&B...")
            sync_cmd = ["uv", "run", "python", str(sync_script), "--output-dir", "models/export"]
            sync_res = run_cmd(sync_cmd, check=False)
            if sync_res.returncode == 0:
                print("  [OK] Artifact synchronization complete. Checkpoints saved to `models/export/`.")
            else:
                print("  [Warning] W&B artifact sync exited with non-zero code. Verify W&B run manually.")
        else:
            print(f"  [Notice] Sync script {sync_script} not found, skipping sync.")

    elapsed = time.time() - start_time
    print_banner(f"Pipeline Completed Successfully in {elapsed:.1f}s")
    print("Summary:")
    print(f"  - Config: {resolved_config.name}")
    print(f"  - Target Session: {target_session or f'Ephemeral {gpu}'}")
    print("  - W&B Tracking: Active (Project: tdc-studio)")
    print("  - Checkpoint Dir: models/export/\n")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Standardized TDC-Studio Cloud Training Pipeline Runner.")
    parser.add_argument("--config", "-c", required=True, help="Path to YAML training configuration")
    parser.add_argument("--session", "-s", default=None, help="Specific Colab session identifier")
    parser.add_argument("--gpu", default="T4", help="GPU accelerator type (T4, L4, A100)")
    parser.add_argument("--epochs", "-e", type=int, default=None, help="Override epoch count")
    parser.add_argument("--skip-dry-run", action="store_true", help="Skip local 1-step dry run validation")
    parser.add_argument("--no-sync", action="store_true", help="Skip W&B model checkpoint synchronization")

    args = parser.parse_args()
    code = execute_pipeline(
        config_path=args.config,
        session=args.session,
        gpu=args.gpu,
        skip_dry_run=args.skip_dry_run,
        no_sync=args.no_sync,
        epochs=args.epochs,
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
