#!/bin/bash
set -euo pipefail


echo "==> Updating system packages..."
apt-get update
apt-get install -y tmux curl unzip

echo "==> Installing uv package manager..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to PATH for current session
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "==> Syncing Python dependencies..."
# uv sync --extra gpu
# uv pip install flash-attn --no-build-isolation

echo "==> Creating data directory..."
mkdir -p ministral/data
mkdir -p ministral/models

# echo "==> Activating virtual environment..."
# source .venv/bin/activate

echo "==> Installing gdown for Google Drive downloads..."
pip install --quiet gdown

echo "==> Downloading training data from Google Drive..."
gdown "https://drive.google.com/uc?id=1pkkfIeVAnD0u4uKWS5jEx1F-fbVZbDLf" 
unzip data.zip -d ministral
rm data.zip

echo ""
echo "==> Setup complete!"
echo ""