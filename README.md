command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

[ -d ".venv" ] || uv venv

uv sync --extra gpu

scp data\news-data-train-v4.jsonl user@example.com:/workspace/text-tune-ai/data
