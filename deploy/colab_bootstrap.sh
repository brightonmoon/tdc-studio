#!/usr/bin/env bash
set -e

# Colab bootstrap script for TDC-Studio
echo "=== [Colab VM] Bootstrapping TDC-Studio GPU Environment ==="
nvidia-smi

# 1. Install uv if not available
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# 2. Sync dependencies using Python 3.11
echo "Syncing dependencies with uv..."
uv sync --extra tdc

# 3. Execute the payload command
TASK_COMMAND="$1"
if [ -z "$TASK_COMMAND" ]; then
    TASK_COMMAND="tdc-studio train --config configs/config.yaml"
fi

echo "Executing: uv run $TASK_COMMAND"
uv run $TASK_COMMAND

echo "=== [Colab VM] Task Completed Successfully ==="
