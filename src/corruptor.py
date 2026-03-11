import json
import os
import pandas as pd
from pydantic import BaseModel
from ollama import Client
from dotenv import load_dotenv
import argparse

load_dotenv()  # Load environment variables from .env file

parser = argparse.ArgumentParser(description="Corrupt German sentences for GEC dataset.")
parser.add_argument("--input", type=str, required=True, help="Path to the input JSONL file.")
parser.add_argument("--output", type=str, required=True, help="Path to the output JSONL file.")
parser.add_argument("--model", type=str, required=False, help="Name of the model to use.", default="gpt-oss:120b-cloud")
args = parser.parse_args()

client = Client(
        host="https://ollama.com",
        headers={'Authorization': 'Bearer ' + os.environ.get('OLLAMA_API_KEY')}
    )

class CorruptionResult(BaseModel):
    original: str
    corrupted: str
    error_types: list[str]
    num_errors: int


SYSTEM_PROMPT = """
You are an expert computational linguist and synthetic data generator specializing in the German language. Your task is to simulate realistic, human-like grammatical errors for a Grammatical Error Correction (GEC) dataset.

I will provide you with a grammatically perfect German sentence. You must return a "corrupted" version of this sentence containing 1 to 5 targeted grammatical errors typical of a B1/B2 level non-native German learner.

CRITICAL CONSTRAINTS (DO NOT VIOLATE):
1. **Preserve Meaning:** The corrupted sentence must retain the exact original semantic meaning. Do not hallucinate new words or remove vital information.
2. **Preserve Tense & Person (No Time Travel):** - NEVER change the tense (e.g., do not change "ging" to "geht"). 
   - NEVER change the pronoun/subject (e.g., do not change "Sie" to "Er").
   - If you corrupt a verb, only mess up the agreement suffix (e.g., "Er machte" -> "Er machten"), but keep it in the same tense.
3. **No Gibberish:** The sentence must remain readable. Do not scramble the word order into pure chaos. Limit word-order errors to minor V2 violations.

TARGET ERROR CATEGORIES (Prioritize these):
- **Case & Gender (Kasus/Genus):** Swap articles incorrectly (e.g., "in dem Haus" -> "in den Haus", "das Auto" -> "der Auto").
- **Missing Articles:** Drop necessary determiners.
- **Adjective Declension:** Use the wrong ending (e.g., "ein schönes Tag" instead of "ein schöner Tag").
- **Orthography:** Lowercase random nouns (e.g., "das haus").
- **Prepositions:** Swap common prepositions incorrectly (e.g., "auf" instead of "an").

OUTPUT FORMAT:
You must respond strictly in JSON format with the following keys:
{
  "original": "The clean sentence provided to you.",
  "corrupted": "The corrupted sentence. Return the full sentence, without **highlighting** the errors.",
  "error_types": ["List", "of", "error", "categories", "applied"],
  "num_errors": "Integer count of errors made"
}
"""

if __name__ == '__main__':
    dataset = pd.read_json(args.input, lines=True)  # Load a sample of the dataset for evaluation
    dataset_dict = dataset.to_dict(orient="records")  # Convert to list of dicts for easier processing

    print(f"Starting to corrupt {len(dataset_dict)} rows...")

    for i, row in enumerate(dataset_dict):
        print(f"Corrupting row {i+1}/{len(dataset_dict)}...")
        response = client.chat(
            messages=[
                {
                    'role': 'system',
                    'content': SYSTEM_PROMPT,
                },
                {
                    'role': 'user',
                    'content': f"Input: {row['cleaned_sentence']}\nOutput:"
                }
            ],
            model=args.model,
            options={
                "temperature": 0.5,
            },
            format=CorruptionResult.model_json_schema(),
        )

        output = response.message.content

        line = json.loads(output)

        with open(args.output, "a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


