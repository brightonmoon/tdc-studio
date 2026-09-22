"""Google Colab CLI orchestrator with multi-account switching and Windows execution support."""

import contextlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Generator, List, Optional

from tdc_studio.core.exceptions import RemoteExecutionError
from tdc_studio.remote.colab_account import ColabAccountManager

QUOTA_ERROR_PATTERNS = [
    r"TooManyAssignmentsError",
    r"Precondition Failed",
    r"ResourceExhausted",
    r"quota exceeded",
    r"cannot assign requested resource",
    r"429 Too Many Requests",
]


class ColabRunner:
    """Orchestrates ephemeral GPU jobs and active session execution on Google Colab."""

    def __init__(
        self,
        gpu_type: str = "T4",
        project_repo: Optional[str] = None,
        account: Optional[str] = None,
        auto_switch_on_quota: bool = False,
        account_manager: Optional[ColabAccountManager] = None,
    ):
        # Normalize GPU name (T4, L4, A100, H100)
        self.gpu_type = gpu_type.upper() if gpu_type else "T4"
        self.project_repo = project_repo
        self.account = account
        self.auto_switch_on_quota = auto_switch_on_quota
        self.account_manager = account_manager or ColabAccountManager()

        # If a specific account was requested at initialization, activate it now
        if self.account:
            self.account_manager.use_account(self.account)

    @staticmethod
    def is_colab_cli_available() -> bool:
        """Check if `colab` CLI executable is installed on host or PATH."""
        return shutil.which("colab") is not None

    @staticmethod
    def is_quota_error(output: str) -> bool:
        """Check if output contains quota limit or assignment error signatures."""
        for pattern in QUOTA_ERROR_PATTERNS:
            if re.search(pattern, output, re.IGNORECASE):
                return True
        return False

    @staticmethod
    @contextlib.contextmanager
    def create_injected_exec_script(
        script_path: str,
        script_args: Optional[List[str]] = None,
        temp_dir: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """Creates a temporary Python wrapper script injecting sys.argv and CLI flags.

        This ports the `colab_exec.ps1` technique to native cross-platform Python.
        Colab CLI's `colab exec` command does not natively pass command line arguments
        to scripts running on remote kernels. By prepending a header that overrides
        `sys.argv` and sets `FORCE_CLI_ARGS=1`, CLI scripts run with full argument support.
        """
        resolved_path = Path(script_path).resolve()
        if not resolved_path.is_file():
            raise FileNotFoundError(f"Target script not found: {resolved_path}")

        original_code = resolved_path.read_text(encoding="utf-8")

        args = script_args or []
        args_repr = repr(args)
        path_repr = repr(str(resolved_path))

        header = (
            "# Auto-generated argument injection wrapper by TDC-Studio ColabRunner\n"
            "import os\n"
            "import sys\n"
            f"sys.argv = [{path_repr}] + {args_repr}\n"
            'os.environ["FORCE_CLI_ARGS"] = "1"\n\n'
        )

        target_dir = Path(temp_dir) if temp_dir else (resolved_path.parent / ".temp_colab")
        target_dir.mkdir(parents=True, exist_ok=True)
        temp_file = target_dir / f"temp_exec_{resolved_path.name}"

        combined_code = header + original_code
        temp_file.write_text(combined_code, encoding="utf-8")

        try:
            yield str(temp_file)
        finally:
            if temp_file.is_file():
                try:
                    temp_file.unlink()
                except OSError:
                    pass

    def build_run_command(
        self,
        command_to_run: str,
        runner_script: str = "deploy/colab_runner_job.py",
        extra_args: Optional[List[str]] = None,
        timeout: float = 3600.0,
    ) -> List[str]:
        """Build the ephemeral `colab run` command line arguments."""
        cmd = ["colab", "run"]
        if timeout:
            cmd.extend(["--timeout", str(timeout)])
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

    def build_exec_command(
        self,
        command_to_run: Optional[str] = None,
        runner_script: str = "deploy/colab_runner_job.py",
        session: Optional[str] = None,
        script_file: Optional[str] = None,
        timeout: Optional[float] = None,
        extra_args: Optional[List[str]] = None,
    ) -> List[str]:
        """Build the `colab exec` command line arguments using valid option flags."""
        target_script = script_file or runner_script
        cmd = ["colab", "exec", "-f", target_script]
        if session:
            cmd.extend(["-s", session])

        if timeout:
            cmd.extend(["--timeout", str(timeout)])

        if extra_args:
            cmd.extend(extra_args)

        return cmd

    def _get_execution_env(self) -> dict:
        """Prepare environment variables with UTF-8 encoding support for Windows."""
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        return env

    def run_remote_job(
        self,
        task_command: str,
        runner_script: str = "deploy/colab_runner_job.py",
        dry_run: bool = False,
        retries: int = 1,
    ) -> int:
        """Execute a training or tuning job on Google Colab Cloud GPU.

        Supports automatic account rotation if quota limits are encountered.
        """
        cmd = self.build_run_command(task_command, runner_script)

        if dry_run:
            return 0

        if not self.is_colab_cli_available():
            raise RemoteExecutionError(
                "Google Colab CLI ('colab') is not installed or not found on PATH. "
                "Install it using: 'uv tool install google-colab-cli' or 'pip install google-colab-cli'."
            )

        env = self._get_execution_env()

        for attempt in range(retries + 1):
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
            )

            # Print stdout and stderr to console
            if process.stdout:
                sys.stdout.write(process.stdout)
            if process.stderr:
                sys.stderr.write(process.stderr)

            if process.returncode == 0:
                return 0

            # Check for quota errors
            combined_output = (process.stdout or "") + (process.stderr or "")
            if self.auto_switch_on_quota and self.is_quota_error(combined_output) and attempt < retries:
                print("\n[ColabRunner] Colab GPU Quota limit exceeded! Attempting account rotation...")
                next_account = self.account_manager.rotate_to_next_account()
                if next_account:
                    print(f"[ColabRunner] Rotated to account '{next_account}'. Retrying job...")
                    continue
                else:
                    print("[ColabRunner] No alternate saved accounts available for rotation.")
                    break
            else:
                return process.returncode

        return process.returncode

    def run_remote_exec(
        self,
        command_to_run: Optional[str] = None,
        runner_script: str = "deploy/colab_runner_job.py",
        session: Optional[str] = None,
        script_file: Optional[str] = None,
        script_args: Optional[List[str]] = None,
        timeout: float = 3600.0,
        dry_run: bool = False,
    ) -> int:
        """Execute a script on an active Colab session with argument injection."""
        target_script = script_file or runner_script
        args = list(script_args) if script_args else []
        if not script_file and command_to_run:
            args.append(command_to_run)
            wandb_key = os.environ.get("WANDB_API_KEY")
            if not wandb_key:
                try:
                    import wandb

                    wandb_key = wandb.Api().api_key
                except Exception:
                    wandb_key = None
            args.append(wandb_key or "none")

        if dry_run:
            cmd = self.build_exec_command(
                script_file=target_script,
                session=session,
                timeout=timeout,
            )
            print(f"[Dry-run] Colab Exec command: {' '.join(cmd)} (Injected args: {args})")
            return 0

        if not self.is_colab_cli_available():
            raise RemoteExecutionError(
                "Google Colab CLI ('colab') is not installed or not found on PATH. "
                "Install it using: 'uv tool install google-colab-cli' or 'pip install google-colab-cli'."
            )

        env = self._get_execution_env()

        with self.create_injected_exec_script(target_script, args) as injected_file:
            cmd = self.build_exec_command(
                script_file=injected_file,
                session=session,
                timeout=timeout,
            )
            process = subprocess.run(cmd, capture_output=False, env=env)
            return process.returncode
