# For Training:

command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

[ -d ".venv" ] || uv venv

uv sync --extra gpu

scp data\news-data-train-v4.jsonl user@example.com:/workspace/text-tune-ai/data

accelerate launch v3/transformers-train-mt5.py

# For Spacy:

uv add pip
python -m spacy download de_core_news_lg
