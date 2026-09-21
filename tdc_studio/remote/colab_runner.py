"""Google Colab CLI orchestrator for cloud GPU training and HPO."""

import os
import shutil
import subprocess
from typing import List, Optional

from tdc_studio.core.exceptions import RemoteExecutionError


class ColabRunner:
    """Orchestrates ephemeral GPU jobs on Google Colab using google-colab-cli."""

    def __init__(self, gpu_type: str = "T4", project_repo: Optional[str] = None):
        # Normalize GPU name (T4, L4, A100, H100)
        self.gpu_type = gpu_type.upper()
        self.project_repo = project_repo

    @staticmethod
    def is_colab_cli_available() -> bool:
        """Check if `colab` CLI executable is installed on host or WSL."""
        return shutil.which("colab") is not None

    def build_run_command(
        self,
        command_to_run: str,
        runner_script: str = "deploy/colab_runner_job.py",
        extra_args: Optional[List[str]] = None,
    ) -> List[str]:
        """Build the ephemeral `colab run` command line arguments."""
        cmd = ["colab", "run"]
        if self.gpu_type:
            cmd.append(f"--gpu={self.gpu_type}")

        if extra_args:
            cmd.extend(extra_args)

        # Local script to upload and execute on Colab VM
        cmd.append(runner_script)
        cmd.append(command_to_run)

        # Forward WANDB_API_KEY if present in environment or netrc
        wandb_key = os.environ.get("WANDB_API_KEY")
        if not wandb_key:
            try:
                import wandb

                wandb_key = wandb.Api().api_key
            except Exception:
                wandb_key = None
        cmd.append(wandb_key or "none")

        return cmd

    def run_remote_job(
        self,
        task_command: str,
        runner_script: str = "deploy/colab_runner_job.py",
        dry_run: bool = False,
    ) -> int:
        """Execute a training or tuning job on Google Colab Cloud GPU."""
        cmd = self.build_run_command(task_command, runner_script)

        if dry_run:
            # For testing without real network / colab execution
            return 0

        if not self.is_colab_cli_available():
            raise RemoteExecutionError(
                "Google Colab CLI ('colab') is not installed or not found on PATH. "
                "Install it using: 'uv tool install google-colab-cli' or 'pip install google-colab-cli'."
            )

        process = subprocess.run(cmd, capture_output=False, text=True)
        return process.returncode
