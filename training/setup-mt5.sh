#!/bin/bash
set -euo pipefail


echo "==> Updating system packages..."
apt-get update
apt-get install -y tmux curl

echo "==> Installing uv package manager..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to PATH for current session
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "==> Syncing Python dependencies..."
uv sync --extra gpu

echo "==> Creating data directory..."
mkdir -p v4/data

echo "==> Activating virtual environment..."
source .venv/bin/activate

echo "==> Installing gdown for Google Drive downloads..."
pip install --quiet gdown

echo "==> Downloading training data from Google Drive..."
gdown "https://drive.google.com/uc?id=1nec3ZsUp5sI5L7pc7t3Au-a880Odi09N" -O v4/data/news-data-v5-train.jsonl
gdown "https://drive.google.com/uc?id=1f1nLcDDnHXQ6rvKmlH-JCNhGeZM_Psv6" -O v4/data/news-data-v5-eval.jsonl

echo ""
echo "==> Setup complete!"
echo ""