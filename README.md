# For Training:

command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

uv sync --extra gpu

scp data\news-data-train-v4.jsonl user@example.com:/workspace/text-tune-ai/data

source .venv/bin/activate

accelerate launch v4/transformers-train-mt5-large.py

# For Spacy:

uv add pip
python -m spacy download de_core_news_lg

# Install repo

cd workspace

git clone https://github.com/Horbee/text-tune-ai.git

cd text-tune-ai

command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

uv sync --extra gpu

mkdir data
cd data

source .venv/bin/activate

pip install gdown
gdown "https://drive.google.com/uc?id=1nec3ZsUp5sI5L7pc7t3Au-a880Odi09N"
gdown "https://drive.google.com/uc?id=1f1nLcDDnHXQ6rvKmlH-JCNhGeZM_Psv6"

# Train

apt-get update
apt-get install -y tmux

source $HOME/.local/bin/env

tmux new -t train
ctrl + b, d

accelerate launch v4/transformers-train-mt5-large.py


scp -r -P 26543 -i ~/.ssh/id_ed25519 root@38.147.83.29:/workspace/text-tune-ai/training/v4/models/ ./
rsync -avz -e "ssh -p 26543 -i ~/.ssh/id_ed25519.pub" root@38.147.83.29:/workspace/text-tune-ai/training/v4/models/ ./
