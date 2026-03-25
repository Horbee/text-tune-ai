import json
import os
import pandas as pd
from pydantic import BaseModel
from ollama import Client
from dotenv import load_dotenv
from src.utils import process_in_batches
import argparse

load_dotenv()  # Load environment variables from .env file

class CorruptionResult(BaseModel):
    original: str
    corrupted: str
    error_types: list[str]
    num_errors: int


SYSTEM_PROMPT = """
You are an expert computational linguist and synthetic data generator specializing in the German language. Your task is to simulate realistic, human-like grammatical errors for a Grammatical Error Correction (GEC) dataset.

I will provide you with a list of grammatically perfect German sentences. You must return a "corrupted" version of each sentence containing 1 to 5 targeted grammatical errors typical of a B1/B2 level non-native German learner.

CRITICAL CONSTRAINTS (DO NOT VIOLATE):
1. **Preserve Meaning & Vocabulary:** The corrupted sentence must retain the exact original semantic meaning. DO NOT add, insert, or invent extra words (like adjectives, adverbs, or nouns) that do not exist in the original sentence. You may only mutate or remove existing words. Do not remove vital information.
2. **Preserve Tense & Person (No Time Travel):** - NEVER change the tense (e.g., do not change "ging" to "geht"). 
   - NEVER change the pronoun/subject (e.g., do not change "Sie" to "Er").
   - If you corrupt a verb, only mess up the agreement suffix (e.g., "Er machte" -> "Er machten"), but keep it in the exact same tense.
3. **No Gibberish:** The sentence must remain readable. Do not scramble the word order into pure chaos. Limit word-order errors to minor V2 violations.

TARGET ERROR CATEGORIES (Prioritize these, but only if the sentence already contains the necessary parts of speech):
- **Subject-Verb Agreement (Kongruenz):** Use the wrong verb ending for the subject, but keep the tense the same (e.g., "Das Kind spielt" -> "Das Kind spielen").
- **Case & Gender (Kasus/Genus):** Swap articles incorrectly (e.g., "in dem Haus" -> "in den Haus", "das Auto" -> "der Auto").
- **Missing Articles:** Drop necessary determiners.
- **Adjective Declension:** Use the wrong ending (e.g., "ein schönes Tag" instead of "ein schöner Tag"). Only apply this if the original sentence already contains an adjective.
- **Orthography:** Lowercase random nouns (e.g., "das haus").
- **Prepositions:** Swap common prepositions incorrectly (e.g., "auf" instead of "an").

OUTPUT FORMAT:
You must respond strictly in JSON format as an array of objects with the following keys:
[
  {
    "original": "The clean sentence provided to you.",
    "corrupted": "The corrupted sentence. Return the full sentence, without **highlighting** the errors.",
    "error_types": ["List", "of", "error", "categories", "applied"],
    "num_errors": "Integer count of errors made"
  }
]
  
INPUT DATA:
[Insert a batch of 10-20 sentences here, separated by newlines]
"""


def corrupt_sentence(sentences: list[str], client: Client, model: str = "gpt-oss:120b-cloud") -> list[CorruptionResult]:
    formatted_input = "\n".join(sentences)

    # print(formatted_input)

    response = client.chat(
        messages=[
            {
                'role': 'system',
                'content': SYSTEM_PROMPT,
            },
            {
                'role': 'user',
                'content': formatted_input,
            }
        ],
        model=model,
        options={
            "temperature": 0.5,
        },
        format=CorruptionResult.model_json_schema(),
    )

    output = response.message.content
    return json.loads(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Corrupt German sentences for GEC dataset.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input JSONL file.")
    parser.add_argument("--output", type=str, required=True, help="Path to the output JSONL file.")
    parser.add_argument("--from_index", type=int, required=False, default=0, help="Starting index for processing.")
    parser.add_argument("--model", type=str, required=False, help="Name of the model to use.", default="gpt-oss:120b-cloud")
    args = parser.parse_args()

    client = Client(
            host="https://ollama.com",
            headers={'Authorization': 'Bearer ' + os.environ.get('OLLAMA_API_KEY')}
        )

    dataset = pd.read_json(args.input, lines=True)[args.from_index:]  # Load a sample of the dataset for evaluation
    all_sentences = dataset["original"].to_list()

    print(f"Starting to corrupt {len(all_sentences)} rows...")

    for current_batch in process_in_batches(all_sentences, batch_size=10):
        batch_result = corrupt_sentence(current_batch, client, model=args.model)

        with open(args.output, "a", encoding="utf-8") as f:
            for line in batch_result:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")


