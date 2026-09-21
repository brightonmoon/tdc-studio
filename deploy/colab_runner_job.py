"""Colab remote execution wrapper executed by `colab run` on remote Colab VM."""

import os
import subprocess
import sys


def main():
    print("=== [Colab Cloud VM] Starting TDC-Studio Task ===")
    task_command = sys.argv[1] if len(sys.argv) > 1 else "tdc-studio train --config configs/config.yaml"
    wandb_key = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "none" else None

    # Set W&B API Key if provided
    if wandb_key:
        os.environ["WANDB_API_KEY"] = wandb_key
        print("W&B API Key configured in remote environment.")

    # 1. Install uv if missing
    res = subprocess.run(["which", "uv"], capture_output=True, text=True)
    if res.returncode != 0:
        print("Installing uv package manager...")
        subprocess.run("curl -LsSf https://astral.sh/uv/install.sh | sh", shell=True, check=True)
        home = os.path.expanduser("~")
        os.environ["PATH"] = f"{home}/.local/bin:" + os.environ.get("PATH", "")

    # 2. Clone repository if not present
    if not os.path.exists("tdc-studio"):
        repo_url = os.environ.get("TDC_STUDIO_REPO", "https://github.com/brightonmoon/tdc-studio.git")
        print(f"Cloning repository from {repo_url}...")
        subprocess.run(["git", "clone", repo_url], check=True)
        os.chdir("tdc-studio")
    else:
        os.chdir("tdc-studio")
        subprocess.run(["git", "pull"], check=False)

    # 3. Sync dependencies using Python 3.11
    print("Syncing Python 3.11 dependencies with uv...")
    subprocess.run(["uv", "sync", "--extra", "tdc"], check=True)

    # 4. Run the requested task command
    print(f"Executing payload: uv run {task_command}")
    cmd_parts = ["uv", "run"] + task_command.split()
    ret = subprocess.run(cmd_parts)
    print("=== [Colab Cloud VM] Task Finished with code:", ret.returncode)
    sys.exit(ret.returncode)


if __name__ == "__main__":
    main()
