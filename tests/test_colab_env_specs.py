"""Unit tests for Colab environment version pinning and runtime verification."""

import sys

from scripts.install_deps import (
    PINNED_PACKAGES,
    get_pip_cmd,
    get_torch_pyg_wheel_url,
    verify_active_environment,
)


def test_pinned_packages_matrix():
    """Verify that critical packages are pinned to stable version specs."""
    assert "PyTDC" in PINNED_PACKAGES
    assert PINNED_PACKAGES["PyTDC"] == "PyTDC==0.4.1"
    assert "torch-geometric" in PINNED_PACKAGES
    assert "rdkit" in PINNED_PACKAGES
    assert "wandb" in PINNED_PACKAGES


def test_verify_active_environment():
    """Verify runtime environment dictionary and python version detection."""
    env = verify_active_environment()
    assert isinstance(env, dict)
    assert "python_version" in env
    assert env["python_version"].startswith(f"{sys.version_info.major}.{sys.version_info.minor}")
    assert "torch" in env
    assert "rdkit" in env


def test_get_pip_cmd():
    """Verify pip command resolution."""
    pip_cmd = get_pip_cmd()
    assert isinstance(pip_cmd, list)
    assert len(pip_cmd) >= 1


def test_get_torch_pyg_wheel_url():
    """Verify PyG wheel URL generation format."""
    url = get_torch_pyg_wheel_url()
    if url is not None:
        assert url.startswith("https://data.pyg.org/whl/")
        assert ".html" in url
