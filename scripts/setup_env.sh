#!/usr/bin/env bash
# Setup script using uv — fast Python package manager
# Run once: bash scripts/setup_env.sh

set -euo pipefail

# Check uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.local/bin/env"
fi

echo "Creating virtual environment with Python 3.11..."
uv venv --python 3.11

echo "Activating and syncing dependencies..."
# shellcheck disable=SC1091
source .venv/bin/activate

echo "Installing core dependencies..."
uv pip install torch transformers datasets trl peft accelerate wandb evaluate \
    scikit-learn pandas jupyter pyyaml bitsandbytes

echo "Installing project in editable mode..."
uv pip install -e .

echo ""
echo "Done. To activate: source .venv/bin/activate"
echo "Copy .env.example to .env and fill in your WANDB credentials."
