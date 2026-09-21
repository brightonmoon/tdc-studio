"""Tests for Google Colab remote orchestrator and notebook generator."""

import os

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
