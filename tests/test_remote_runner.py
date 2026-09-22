"""Tests for Google Colab remote orchestrator, argument injection, and notebook generator."""

import os
from pathlib import Path

from tdc_studio.remote.colab_account import ColabAccountManager
from tdc_studio.remote.colab_runner import ColabRunner
from tdc_studio.remote.notebook_gen import export_notebook_file, generate_colab_notebook


def test_colab_runner_build_command():
    runner = ColabRunner(gpu_type="a100")
    cmd = runner.build_run_command("tdc-studio train --config configs/config.yaml")

    assert cmd[0] == "colab"
    assert cmd[1] == "run"
    assert "--gpu=A100" in cmd
    assert any("colab_runner_job.py" in arg for arg in cmd)


def test_colab_runner_dry_run():
    runner = ColabRunner(gpu_type="t4")
    code = runner.run_remote_job("tdc-studio tune --config configs/config.yaml", dry_run=True)
    assert code == 0


def test_colab_runner_build_exec_command():
    runner = ColabRunner()
    cmd = runner.build_exec_command("tdc-studio train --config configs/config.yaml")

    assert cmd[0] == "colab"
    assert cmd[1] == "exec"
    assert any("colab_runner_job.py" in arg for arg in cmd)


def test_colab_runner_exec_dry_run():
    runner = ColabRunner()
    code = runner.run_remote_exec(
        command_to_run="tdc-studio train --config configs/config.yaml", dry_run=True
    )
    assert code == 0


def test_colab_runner_injected_exec_script(tmp_path):
    # Create a dummy python script
    dummy_script = tmp_path / "train_script.py"
    dummy_script.write_text("print('Executing original script')", encoding="utf-8")

    script_args = ["--epochs", "10", "--lr", "0.001"]
    temp_dir = tmp_path / "temp_injected"

    injected_path_str = None
    with ColabRunner.create_injected_exec_script(
        str(dummy_script), script_args=script_args, temp_dir=str(temp_dir)
    ) as injected_file:
        injected_path = Path(injected_file)
        injected_path_str = injected_file
        assert injected_path.is_file()

        content = injected_path.read_text(encoding="utf-8")
        assert "sys.argv = " in content
        assert "--epochs" in content
        assert "--lr" in content
        assert 'os.environ["FORCE_CLI_ARGS"] = "1"' in content
        assert "print('Executing original script')" in content

    # Check cleanup
    assert not Path(injected_path_str).exists()


def test_colab_runner_quota_error_detection():
    err1 = "TooManyAssignmentsError: Failed to issue request POST ... Precondition Failed"
    err2 = "ResourceExhausted: Colab GPU quota limit exceeded for T4"
    err3 = "Everything finished normally without errors."

    assert ColabRunner.is_quota_error(err1) is True
    assert ColabRunner.is_quota_error(err2) is True
    assert ColabRunner.is_quota_error(err3) is False


def test_colab_runner_account_selection(tmp_path):
    mgr = ColabAccountManager(config_dir=tmp_path)
    mgr.token_path.write_bytes(b'{"token": "initial"}')
    mgr.save_account("account_1")

    mgr.token_path.write_bytes(b'{"token": "secondary"}')
    mgr.save_account("account_2")

    # Initializing ColabRunner with account="account_1" should activate it
    _ = ColabRunner(account="account_1", account_manager=mgr)
    assert mgr.get_active_account() == "account_1"


def test_generate_colab_notebook(tmp_path):
    nb = generate_colab_notebook(
        repo_url="https://github.com/example/tdc-studio.git",
        run_command="tdc-studio train",
    )
    assert nb["metadata"]["accelerator"] == "GPU"
    assert len(nb["cells"]) >= 4

    out_file = os.path.join(tmp_path, "test_colab.ipynb")
    export_notebook_file(out_file, run_command="tdc-studio train")
    assert os.path.exists(out_file)
    assert os.path.getsize(out_file) > 100


def test_colab_runner_create_bundle():
    b64 = ColabRunner.create_bundle_b64()
    assert isinstance(b64, str)
    assert len(b64) > 1000  # Should be at least tens of KB


def test_colab_runner_injected_run_script(tmp_path):
    dummy_runner = tmp_path / "dummy_job.py"
    dummy_runner.write_text("print('Dummy runner')", encoding="utf-8")
    temp_dir = tmp_path / "temp_run"

    with ColabRunner.create_injected_run_script(
        str(dummy_runner), bundle_b64="test_b64_content", temp_dir=str(temp_dir)
    ) as injected_file:
        p = Path(injected_file)
        assert p.is_file()
        content = p.read_text(encoding="utf-8")
        assert 'BUNDLE_B64 = "test_b64_content"' in content
        assert "print('Dummy runner')" in content

    assert not Path(injected_file).exists()

