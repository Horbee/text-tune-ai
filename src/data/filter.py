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
    cleaned_sentence: str
    keep: bool
    reason: str


SYSTEM_PROMPT = """
You are an expert German linguist and NLP data curator. Your task is to filter and clean a dataset of raw German text to select only high-quality sentences suitable as "ground truth" targets for training a Grammatical Error Correction (GEC) model.

I will provide you with a list of raw German sentences. For each sentence, you must perform two steps:

STEP 1: TEXT CLEANING
- Remove unnecessary formatting, HTML tags, markdown symbols, and weird unicode characters.
- Normalize whitespace (remove double spaces, leading/trailing spaces).
- Do NOT alter the core wording or fix minor grammatical errors. If a sentence is fundamentally broken or heavily erroneous, do not fix it; it will be filtered out in Step 2.

STEP 2: EVALUATION & FILTERING
Evaluate the cleaned sentence based on its value for teaching German grammar. 
Mark the sentence to be KEPT (keep: true) ONLY IF it meets all the following criteria:
1. Length & Complexity: It has at least 5 words and contains a complete thought (usually subject, verb, and object/complement).
2. Grammatical Value: It demonstrates proper German sentence structure (e.g., correct verb-second placement, proper case usage, or valid subordinate clauses). 
3. Coherence: It makes logical sense and is not a random string of words.
4. Language: It is definitively German (reject mixed language or pure English sentences).

Mark the sentence to be DISCARDED (keep: false) if it is:
- A short, low-value fragment (e.g., "Ja, genau.", "Hallo zusammen!").
- Overly conversational slang with no structural value.
- Heavily ungrammatical or gibberish.
- Primarily numbers, URLs, or lists without grammatical connective tissue.

OUTPUT FORMAT:
Provide the output strictly as a JSON array of objects. Do not include any conversational filler. Each object must have the following structure:
[
  {
    "original_sentence": "The exact input string",
    "cleaned_sentence": "The cleaned text",
    "keep": true/false,
    "reason": "A brief 3-5 word explanation for the decision"
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
            "temperature": 0.5,
        },
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


    dataset = pd.read_json(args.input, lines=True)[args.from_index: ]  # Load a sample of the dataset for evaluation
    all_sentences = dataset["text"].to_list()

    print(f"Starting to filter {len(all_sentences)} rows...")

    for current_batch in process_in_batches(all_sentences, batch_size=10):
        filter_results = filter_sentences(current_batch, client, model=args.model)
        batch_result = [row for row in filter_results if row["keep"]]  # Keep only sentences marked for retention

        with open(args.output, "a", encoding="utf-8") as f:
            for line in batch_result:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")



