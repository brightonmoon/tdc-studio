"""Google Colab CLI orchestrator for cloud GPU training and HPO."""

import shutil
import subprocess
from typing import List, Optional

from tdc_studio.core.exceptions import RemoteExecutionError


class ColabRunner:
    """Orchestrates ephemeral GPU jobs on Google Colab using google-colab-cli."""

    def __init__(self, gpu_type: str = "t4", project_repo: Optional[str] = None):
        self.gpu_type = gpu_type.lower()
        self.project_repo = project_repo

    @staticmethod
    def is_colab_cli_available() -> bool:
        """Check if `colab` CLI executable is installed on host or WSL."""
        return shutil.which("colab") is not None

    def build_run_command(
        self,
        command_to_run: str,
        bootstrap_script: str = "deploy/colab_bootstrap.sh",
        extra_args: Optional[List[str]] = None,
    ) -> List[str]:
        """Build the ephemeral `colab run` command line arguments."""
        cmd = ["colab", "run", f"--gpu={self.gpu_type}"]
        if extra_args:
            cmd.extend(extra_args)

        # The script to run inside Colab VM
        script_payload = f'bash {bootstrap_script} "{command_to_run}"'
        cmd.append(script_payload)
        return cmd

    def run_remote_job(
        self,
        task_command: str,
        bootstrap_script: str = "deploy/colab_bootstrap.sh",
        dry_run: bool = False,
    ) -> int:
        """Execute a training or tuning job on Google Colab Cloud GPU."""
        cmd = self.build_run_command(task_command, bootstrap_script)

        if dry_run:
            # For testing without real network / colab execution
            return 0

        if not self.is_colab_cli_available():
            raise RemoteExecutionError(
                "Google Colab CLI ('colab') is not installed or not found on PATH. "
                "Install it using: 'uv tool install google-colab-cli' or 'pip install google-colab-cli'. "
                "(Note: If running on Windows, run inside WSL2)."
            )

        process = subprocess.run(cmd, capture_output=False, text=True)
        return process.returncode
