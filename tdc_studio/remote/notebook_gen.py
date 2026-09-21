"""Generate standalone Google Colab Notebook (.ipynb) for cloud GPU execution."""

import json
from typing import Any, Dict


def generate_colab_notebook(
    repo_url: str = "https://github.com/your-org/tdc-studio.git",
    run_command: str = "tdc-studio train --config configs/config.yaml",
) -> Dict[str, Any]:
    """Create Jupyter notebook structure configured for Google Colab GPU training."""
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# TDC-Studio Cloud GPU Training Runner\n",
                "This notebook provisions the environment using `uv` and executes training on Colab GPU.",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Check GPU availability\n",
                "!nvidia-smi\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 1. Install uv for ultra-fast package management\n",
                "!curl -LsSf https://astral.sh/uv/install.sh | sh\n",
                "import os\n",
                "os.environ['PATH'] = f\"{os.path.expanduser('~')}/.local/bin:\" + os.environ['PATH']\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 2. Clone repository & install dependencies\n",
                f"!git clone {repo_url} || true\n",
                "%cd tdc-studio\n",
                "!uv sync --extra tdc\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 3. Execute cloud training or HPO\n",
                f"!uv run {run_command}\n",
            ],
        },
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    return notebook


def export_notebook_file(
    output_path: str,
    repo_url: str = "https://github.com/your-org/tdc-studio.git",
    run_command: str = "tdc-studio train --config configs/config.yaml",
) -> str:
    """Save the Colab notebook to disk."""
    nb = generate_colab_notebook(repo_url, run_command)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    return output_path
