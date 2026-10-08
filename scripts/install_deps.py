"""Robust, version-pinned dependency installer and environment verifier for Google Colab VM.

Supports dynamic Python 3.10/3.11 runtime detection, PyTorch CUDA wheel resolution,
and idempotent fast-path verification for zero overhead on already-configured VMs.
"""

from __future__ import annotations

import importlib
import platform
import shutil
import subprocess
import sys

PINNED_PACKAGES = {
    "PyTDC": "PyTDC==0.4.1",
    "rdkit": "rdkit>=2023.9.1",
    "scikit-learn": "scikit-learn>=1.3.0,<1.6.0",
    "catboost": "catboost>=1.2.0",
    "lightgbm": "lightgbm>=4.0.0,<4.5.0",
    "wandb": "wandb>=0.16.0",
    "polite-http": "polite-http",
    "scipy": "scipy>=1.10.0,<1.15.0",
    "pandas": "pandas>=2.0.0,<2.3.0",
    "torch-geometric": "torch-geometric>=2.4.0",
}


def get_pip_cmd() -> list[str]:
    """Resolve pip runner executable (python -m pip, uv pip, or pip)."""
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return [sys.executable, "-m", "pip"]
    except Exception:
        pass

    if shutil.which("uv"):
        return ["uv", "pip"]
    if shutil.which("pip"):
        return ["pip"]
    return [sys.executable, "-m", "pip"]


def run_cmd(cmd: list[str]) -> bool:
    """Execute command with piped output."""
    try:
        subprocess.check_call(cmd)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}: {' '.join(cmd)}")
        return False


def get_torch_pyg_wheel_url() -> str | None:
    """Resolve compatible PyTorch Geometric wheel repository URL based on active PyTorch and CUDA."""
    try:
        import torch

        torch_ver = torch.__version__.split("+")[0]
        # Get major.minor e.g. 2.1, 2.2, 2.3, 2.4, 2.5
        parts = torch_ver.split(".")
        if len(parts) >= 2:
            base_torch = f"{parts[0]}.{parts[1]}.0"
        else:
            base_torch = torch_ver

        cuda_ver = torch.version.cuda
        if cuda_ver:
            cuda_str = "cu" + cuda_ver.replace(".", "")
        else:
            cuda_str = "cpu"

        wheel_url = f"https://data.pyg.org/whl/torch-{base_torch}+{cuda_str}.html"
        return wheel_url
    except Exception:
        return None


def verify_active_environment() -> dict[str, str | bool]:
    """Check whether all critical ADMET/DTI libraries can be cleanly imported."""
    status: dict[str, str | bool] = {}

    status["python_version"] = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )

    modules_to_test = [
        ("torch", None),
        ("torch_geometric", None),
        ("rdkit", None),
        ("tdc", None),
        ("sklearn", None),
        ("lightgbm", None),
        ("wandb", None),
    ]

    all_ok = True
    for mod_name, _ in modules_to_test:
        try:
            mod = importlib.import_module(mod_name)
            ver = getattr(mod, "__version__", "available")
            status[mod_name] = ver
        except ImportError:
            status[mod_name] = False
            all_ok = False

    status["all_healthy"] = all_ok

    try:
        import torch

        status["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            status["cuda_device"] = torch.cuda.get_device_name(0)
    except Exception:
        status["cuda_available"] = False

    return status


def main():
    print("=" * 80)
    print("=== Colab VM Dynamic Environment & Pinned Dependencies Installer ===")
    print("=" * 80)
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"Active Runtime: Python {py_ver} on {platform.system()} ({platform.machine()})")

    if sys.version_info.minor not in (10, 11, 12):
        print(f"Warning: Untested Python version 3.{sys.version_info.minor}. Recommended: 3.10 or 3.11.")

    # 1. Fast-path check: Idempotent skip if all libraries already healthy
    print("\n[Step 1/3] Checking existing environment health...")
    env_info = verify_active_environment()
    if env_info.get("all_healthy"):
        print("[OK] All required dependencies are already installed and verified! Skipping pip install.")
        print("Environment Summary:")
        for k, v in env_info.items():
            print(f"  - {k}: {v}")
        print("=" * 80)
        return

    # 2. PyG & Wheel Resolution
    print("\n[Step 2/3] Installing pinned packages...")
    pyg_wheel_url = get_torch_pyg_wheel_url()
    if pyg_wheel_url:
        print(f"Detected PyTorch configuration. PyG wheel index: {pyg_wheel_url}")

    install_list = list(PINNED_PACKAGES.values())
    pip_base = get_pip_cmd()

    pip_cmd = pip_base + ["install", "--upgrade"]
    if pyg_wheel_url:
        pip_cmd.extend(["-f", pyg_wheel_url])
    pip_cmd.extend(install_list)

    print(f"Executing: {' '.join(pip_cmd[:8])} ... ({len(install_list)} packages)")
    success = run_cmd(pip_cmd)
    if not success:
        print("Primary install command encountered warnings/errors. Attempting individual fallback...")
        for pkg in install_list:
            run_cmd(pip_base + ["install", pkg])

    # 3. Final Verification
    print("\n[Step 3/3] Final health verification...")
    final_info = verify_active_environment()
    print("Verification Results:")
    for k, v in final_info.items():
        print(f"  - {k}: {v}")

    if final_info.get("all_healthy"):
        print("\n[*] Environment setup and dependency pin verified successfully!")
    else:
        print("\n[!] Warning: Some packages are not yet available or failed to import.")

    print("=" * 80)


if __name__ == "__main__":
    main()

