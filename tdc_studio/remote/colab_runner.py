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


def _safe_stream_write(stream, text: str) -> None:
    """Safely write text to a stream handling Windows cp949/UnicodeEncodeError."""
    try:
        stream.write(text)
    except UnicodeEncodeError:
        encoding = getattr(stream, "encoding", None) or "utf-8"
        sanitized = text.encode(encoding, errors="replace").decode(encoding)
        stream.write(sanitized)
    except Exception:
        pass
    with contextlib.suppress(Exception):
        stream.flush()


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

    @classmethod
    @contextlib.contextmanager
    def create_injected_exec_script(
        cls,
        script_path: str,
        script_args: Optional[List[str]] = None,
        bundle_b64: Optional[str] = None,
        temp_dir: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """Creates a temporary Python wrapper script injecting sys.argv, CLI flags, and workspace bundle.

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
        b64_str = bundle_b64 if bundle_b64 is not None else cls.create_bundle_b64()

        header = (
            "# Auto-generated argument injection wrapper by TDC-Studio ColabRunner\n"
            f'BUNDLE_B64 = "{b64_str}"\n'
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

    @classmethod
    def create_bundle_b64(cls, workspace_root: Optional[Path] = None) -> str:
        """Pack core project code (tdc_studio, configs, pyproject.toml, README.md) into base64 zip."""
        import base64
        import io
        import zipfile

        root = (workspace_root or Path.cwd()).resolve()
        targets = ["tdc_studio", "configs", "pyproject.toml", "README.md"]
        buf = io.BytesIO()

        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for t in targets:
                tp = root / t
                if tp.is_file():
                    zf.write(tp, arcname=t)
                elif tp.is_dir():
                    for item in tp.rglob("*"):
                        if "__pycache__" in item.parts or item.suffix in (
                            ".pyc",
                            ".pt",
                            ".pth",
                            ".log",
                        ):
                            continue
                        if item.is_file():
                            rel_path = item.relative_to(root)
                            zf.write(item, arcname=str(rel_path).replace("\\", "/"))

        return base64.b64encode(buf.getvalue()).decode("utf-8")

    @classmethod
    @contextlib.contextmanager
    def create_injected_run_script(
        cls,
        runner_script: str = "deploy/colab_runner_job.py",
        bundle_b64: Optional[str] = None,
        temp_dir: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """Inject BUNDLE_B64 into runner script so Colab runs the exact local workspace."""
        resolved_path = Path(runner_script).resolve()
        if not resolved_path.is_file():
            raise FileNotFoundError(f"Runner script not found: {resolved_path}")

        original_code = resolved_path.read_text(encoding="utf-8")

        b64_str = bundle_b64 if bundle_b64 is not None else cls.create_bundle_b64()
        header = f'BUNDLE_B64 = "{b64_str}"\n\n'

        target_dir = Path(temp_dir) if temp_dir else (resolved_path.parent / ".temp_colab")
        target_dir.mkdir(parents=True, exist_ok=True)
        temp_file = target_dir / f"run_{resolved_path.name}"

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
        env["PYTHONUTF8"] = "1"
        return env

    def run_remote_job(
        self,
        task_command: str,
        runner_script: str = "deploy/colab_runner_job.py",
        dry_run: bool = False,
        retries: int = 1,
        embed_bundle: bool = True,
    ) -> int:
        """Execute a training or tuning job on Google Colab Cloud GPU.

        Supports automatic account rotation if quota limits are encountered.
        """
        if dry_run:
            cmd = self.build_run_command(task_command, runner_script)
            print(f"[Dry-run] Colab Run command: {' '.join(cmd)}")
            return 0

        if not self.is_colab_cli_available():
            raise RemoteExecutionError(
                "Google Colab CLI ('colab') is not installed or not found on PATH. "
                "Install it using: 'uv tool install google-colab-cli' or 'pip install google-colab-cli'."
            )

        env = self._get_execution_env()

        for attempt in range(retries + 1):
            accumulated_output: List[str] = []
            if embed_bundle:
                script_ctx = self.create_injected_run_script(runner_script)
            else:
                script_ctx = contextlib.nullcontext(runner_script)

            with script_ctx as target_script:
                cmd = self.build_run_command(task_command, target_script)

                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    env=env,
                )

                if process.stdout:
                    for line in iter(process.stdout.readline, ""):
                        _safe_stream_write(sys.stdout, line)
                        accumulated_output.append(line)
                    process.stdout.close()

                returncode = process.wait()

            if returncode == 0:
                return 0

            # Check for quota errors
            full_log = "".join(accumulated_output)
            if self.auto_switch_on_quota and self.is_quota_error(full_log) and attempt < retries:
                print(
                    "\n[ColabRunner] Colab GPU Quota limit exceeded! Attempting account rotation...",
                    flush=True,
                )
                next_account = self.account_manager.rotate_to_next_account()
                if next_account:
                    print(
                        f"[ColabRunner] Rotated to account '{next_account}'. Retrying job...",
                        flush=True,
                    )
                    continue
                else:
                    print(
                        "[ColabRunner] No alternate saved accounts available for rotation.",
                        flush=True,
                    )
                    break
            else:
                return returncode

        return returncode

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
