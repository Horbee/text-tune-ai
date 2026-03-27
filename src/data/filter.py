import json
import os
import pandas as pd
from pydantic import BaseModel
from ollama import Client
from dotenv import load_dotenv
from src.utils import process_in_batches
import argparse

load_dotenv()  # Load environment variables from .env file

class FilterResult(BaseModel):
    original_sentence: str
    linguistic_analysis: str
    cleaned_sentence: str
    keep: bool


SYSTEM_PROMPT = """
You are an expert German linguist and NLP data curator. Your task is to filter, clean, and verify a dataset of raw German text to select ONLY absolutely flawless sentences suitable as "ground truth" targets for training a Grammatical Error Correction (GEC) model.

I will provide you with a list of raw German sentences. For each sentence, perform the following steps:

STEP 1: TEXT CLEANING & MINOR CORRECTIONS
- Remove unnecessary formatting, HTML tags, markdown symbols, and weird unicode characters.
- Normalize whitespace (remove double spaces, leading/trailing spaces).
- FIX objective, minor grammatical, punctuation, or spelling errors (e.g., correct a wrong case, fix subject-verb agreement, add a missing comma). 
- CRITICAL: Do NOT alter the author's core vocabulary, style, or phrasing. Only make the absolute minimum edits required to make the sentence 100% grammatically perfect.

STEP 2: EVALUATION & FILTERING
Evaluate the resulting `cleaned_sentence`. Mark it to be KEPT (keep: true) ONLY IF it meets ALL the following criteria:
1. Flawless Grammar: The `cleaned_sentence` is now 100% grammatically correct in standard High German (Hochdeutsch).
2. Length & Complexity: It has at least 5 words and contains a complete, self-contained thought (subject, verb, object/complement).
3. Coherence: It makes logical semantic sense and is not gibberish.
4. Language: It is definitively German (reject mixed language or pure English sentences).

Mark the sentence to be DISCARDED (keep: false) if it is:
- A short, low-value fragment (e.g., "Ja, genau.", "Hallo zusammen!").
- Overly conversational slang where "correct" grammar is subjective.
- So heavily broken or erroneous in the original text that fixing it required entirely rewriting the semantic meaning.
- Primarily numbers, URLs, or lists.

OUTPUT FORMAT:
Provide the output strictly as a JSON array of objects. Do not include any conversational filler. The structure MUST follow this exact order to allow for reasoning:
[
  {
    "original_sentence": "The exact input string",
    "linguistic_analysis": "A brief 1-2 sentence analysis identifying any errors in the original text and verifying if the sentence meets all criteria for a GEC ground truth target.",
    "cleaned_sentence": "The flawless, cleaned text (or empty string if discarded)",
    "keep": true/false
  }
]

INPUT DATA:
[Insert a batch of 10-20 sentences here, separated by newlines]
"""


def filter_sentences(sentences: list[str], client: Client, model: str = "gpt-oss:120b-cloud") -> list[FilterResult]:
    formatted_input = "\n".join(sentences)

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
            "temperature": 0.0,
        },
        think="high",
        format=FilterResult.model_json_schema(),
    )

    output = response.message.content
    return json.loads(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Filter German sentences for GEC dataset.")
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
    all_sentences = dataset["text"].to_list()

    print(f"Starting to filter {len(all_sentences)} rows...")

    for current_batch in process_in_batches(all_sentences, batch_size=10):
        filter_results = filter_sentences(current_batch, client, model=args.model)
        # batch_result = [row for row in filter_results if row["keep"]]  # Keep only sentences marked for retention

        with open(args.output, "a", encoding="utf-8") as f:
            for line in filter_results:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")



