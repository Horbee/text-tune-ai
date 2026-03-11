import json
import os
import pandas as pd
from pydantic import BaseModel
from ollama import Client
from dotenv import load_dotenv
import argparse

load_dotenv()  # Load environment variables from .env file

parser = argparse.ArgumentParser(description="Filter German sentences for GEC dataset.")
parser.add_argument("--input", type=str, required=True, help="Path to the input JSONL file.")
parser.add_argument("--output", type=str, required=True, help="Path to the output JSONL file.")
parser.add_argument("--model", type=str, required=False, help="Name of the model to use.", default="gpt-oss:120b-cloud")
args = parser.parse_args()

client = Client(
        host="https://ollama.com",
        headers={'Authorization': 'Bearer ' + os.environ.get('OLLAMA_API_KEY')}
    )

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

if __name__ == '__main__':
    dataset = pd.read_json(args.input, lines=True)  # Load a sample of the dataset for evaluation
    dataset_dict = dataset.to_dict(orient="records")  # Convert to list of dicts for easier processing

    print(f"Starting to filter {len(dataset_dict)} rows...")

    for i in range(0, len(dataset_dict), 10):  # Process in batches of 10
        batch = dataset_dict[i:i+10]
        print(f"Processing batch {i//10 + 1} of {int(len(dataset_dict) / 10)} batches...")

        formatted_input = "\n".join([row['label'] for row in batch])  # Assuming 'label' column has the sentences

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
            model=args.model,
            options={
                "temperature": 0.5,
            },
            format=FilterResult.model_json_schema(),
        )

        output = response.message.content

        batch_result = json.loads(output)

        with open(args.output, "a", encoding="utf-8") as f:
            for line in batch_result:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")


