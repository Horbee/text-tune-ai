from src.inference import correct_text_latest
import pandas as pd
import json
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(
    prog="eval_create_corrections",
    description="Create correction pairs using an Ollama model for Grammatical Error Correction (GEC)",
    usage="python eval_create_corrections.py ministral-3:8b",
)

parser.add_argument(
    "--model",
    "-m",
    type=str,
    required=True,
    help="Name of the Ollama model to use, e.g., ministral-3:8b",
)
parser.add_argument(
    "--input",
    "-i",
    type=str,
    required=True,
    help="Path to the input JSONL file, e.g., train-v7.jsonl",
)
args = parser.parse_args()

df = pd.read_json(args.input, lines=True)
safe_model_name = args.model.replace(":", "-")
output_file = f"{safe_model_name}-corrected.jsonl"

for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
    original_text = row["original"]
    user_input = row["corrupted"]

    try:
        corrected_text = correct_text_latest(args.model, user_input)

        # Append single result to file (much more efficient than rewriting entire file)
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "corrupted": user_input,
                        "model_corrected": corrected_text,
                        "original": original_text,
                    },
                    ensure_ascii=False,
                )
            )
            f.write("\n")

    except Exception as e:
        print(f"Error processing row {index}: {e}")
