"""Colab remote execution wrapper executed by `colab run` on remote Colab VM."""

import os
import subprocess
import sys


def run_command_streaming(cmd, shell: bool = False) -> int:
    """Execute command and stream stdout/stderr line-by-line so Jupyter captures it."""
    process = subprocess.Popen(
        cmd,
        shell=shell,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    if process.stdout:
        for line in iter(process.stdout.readline, ""):
            print(line, end="", flush=True)
        process.stdout.close()
    return process.wait()


def main():
    os.environ["TDC_REMOTE_EXECUTION"] = "1"
    print("=== [Colab Cloud VM] Starting TDC-Studio Task ===", flush=True)
    task_command = sys.argv[1] if len(sys.argv) > 1 else "tdc-studio train --config configs/config.yaml"
    wandb_key = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "none" else None

    # Set W&B API Key if provided
    if wandb_key:
        os.environ["WANDB_API_KEY"] = wandb_key
        print("W&B API Key configured in remote environment.", flush=True)

    # 1. Ensure ~/.local/bin is in PATH and install uv if missing
    home = os.path.expanduser("~")
    local_bin = f"{home}/.local/bin"
    if local_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"{local_bin}:" + os.environ.get("PATH", "")

    res = subprocess.run(["which", "uv"], capture_output=True, text=True)
    if res.returncode != 0:
        print("Installing uv package manager...", flush=True)
        run_command_streaming("curl -LsSf https://astral.sh/uv/install.sh | sh", shell=True)

    # 2. Extract bundled code if available, or clone repository
    bundle_b64 = globals().get("BUNDLE_B64", None)
    workspace_dir = os.path.abspath("tdc-studio")
    if bundle_b64:
        import base64
        import io
        import zipfile

        print("Extracting bundled local workspace...", flush=True)
        data = base64.b64decode(bundle_b64)
        with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
            zf.extractall(workspace_dir)
        os.chdir(workspace_dir)
    elif not os.path.exists(workspace_dir):
        repo_url = os.environ.get("TDC_STUDIO_REPO", "https://github.com/brightonmoon/tdc-studio.git")
        print(f"Cloning repository from {repo_url}...", flush=True)
        run_command_streaming(["git", "clone", repo_url, workspace_dir])
        os.chdir(workspace_dir)
    else:
        os.chdir(workspace_dir)
        run_command_streaming(["git", "pull"])

    # 3. Sync dependencies using uv
    print("Syncing Python dependencies with uv...", flush=True)
    sync_code = run_command_streaming(["uv", "sync", "--extra", "tdc"])
    if sync_code != 0:
        print(f"[Warning] 'uv sync --extra tdc' exited with {sync_code}. Retrying core 'uv sync'...", flush=True)
        run_command_streaming(["uv", "sync"])

    # Ensure setuptools<72 for PyTDC pkg_resources compatibility
    run_command_streaming(["uv", "pip", "install", "setuptools<72"])

    # 4. Run the requested task command with live streaming output
    if task_command.startswith("uv run "):
        cmd_parts = task_command.split()
    else:
        cmd_parts = ["uv", "run"] + task_command.split()
    print(f"Executing payload: {' '.join(cmd_parts)}", flush=True)
    retcode = run_command_streaming(cmd_parts)
    print("=== [Colab Cloud VM] Task Finished with code:", retcode, flush=True)
    if retcode != 0:
        sys.exit(retcode)


if __name__ == "__main__":
    main()
