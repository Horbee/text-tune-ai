import json
import os
from ollama import Client
from pydantic import BaseModel
from dotenv import load_dotenv
import pandas as pd
import argparse
from utils import process_in_batches


load_dotenv()


class EvalResult(BaseModel):
    original_sentence: str
    model_output: str
    is_grammatically_correct: bool
    meaning_preserved: bool
    reason: str


SYSTEM_PROMPT = """
You are an expert German linguist evaluating a Grammatical Error Correction (GEC) system.

Evaluate the MODEL OUTPUT on two criteria:
1. is_grammatically_correct: Is the MODEL OUTPUT completely free of grammatical, spelling, case (Kasus), and punctuation errors? (true/false)
2. meaning_preserved: Does the MODEL OUTPUT preserve the exact intended meaning of the ORIGINAL sentence without deleting crucial information or hallucinating new facts? (true/false)


OUTPUT FORMAT:
Provide the output strictly as a JSON array of objects. Do not include any conversational filler. Each object must have the following structure:
[
  {
    "original_sentence": "The exact input string of the original sentence",
    "model_output": "The exact input string of the model's corrected output",
    "is_grammatically_correct": true/false,
    "meaning_preserved": true/false,
    "reason": "A brief 3-5 word explanation for the decision"
  }
]

INPUT DATA:
[Insert a batch of 10-20 sentence pairs here, in the format of ORIGINAL: "original sentence" MODEL OUTPUT: "model output", separated by newlines]
"""


def evaluate_correction(sentences: list[tuple[str, str]], client: Client, model: str = "gpt-oss:120b-cloud") -> list[EvalResult]:
    formatted_input = "\n".join([f"ORIGINAL: \"{orig}\" MODEL OUTPUT: \"{output}\"" for orig, output in sentences])

    response = client.chat(
        model=model,
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
        options={'temperature': 0.0}, # Temperature 0 for maximum consistency
        format=EvalResult.model_json_schema(),
    )
    
    result_str = response['message']['content']
    return json.loads(result_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Try out a text correction model')
    parser.add_argument('correction_file', type=str, help='Path to the correction file, e.g., ministral-3-14b-corrected.jsonl')
    parser.add_argument("--model", type=str, required=False, help="Name of the model to use.", default="gpt-oss:120b-cloud")
    parser.add_argument("--from_index", type=int, required=False, default=0, help="Starting index for processing.")
    args = parser.parse_args()

    OUTPUT_FILE = f"{args.correction_file}-results.jsonl"

    client = Client(
        host="https://ollama.com",
        headers={'Authorization': 'Bearer ' + os.environ.get('OLLAMA_API_KEY')}
    )

    eval_data = pd.read_json(args.correction_file, lines=True)[args.from_index:] 
    sentence_paris = list(zip(eval_data["original"].to_list(), eval_data["model_corrected"].to_list()))
    
    print(f"Starting to evaluate {len(sentence_paris)} rows...")

    results = []
    strict_correct_count = 0
    grammar_correct_count = 0
    meaning_preserved_count = 0

    for current_batch in process_in_batches(sentence_paris, batch_size=10):
        batch_results = evaluate_correction(current_batch, client, model=args.model)
        for judgment in batch_results:
            if judgment["is_grammatically_correct"]:
                grammar_correct_count += 1
            if judgment["meaning_preserved"]:
                meaning_preserved_count += 1
            if judgment["is_grammatically_correct"] and judgment["meaning_preserved"]:
                strict_correct_count += 1

            results.append({
                "original": judgment["original_sentence"],
                "model_corrected": judgment["model_output"],
                "is_grammatically_correct": judgment["is_grammatically_correct"],
                "meaning_preserved": judgment["meaning_preserved"],
                "reason": judgment["reason"]
            })

            with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(results[-1], ensure_ascii=False) + "\n")

    # ---------------------------------------------------------
    # CALCULATE METRICS
    # ---------------------------------------------------------
    total = len(eval_data)
    strict_acc = (strict_correct_count / total) * 100
    grammar_acc = (grammar_correct_count / total) * 100
    meaning_acc = (meaning_preserved_count / total) * 100

    print("\n" + "="*40)
    print(" 🏆 GEC EVALUATION REPORT")
    print("="*40)
    print(f"Total Sentences Evaluated : {total}")
    print(f"1. Grammar Score          : {grammar_acc:.1f}%  (Sentences with perfect grammar)")
    print(f"2. Meaning Score          : {meaning_acc:.1f}%  (Sentences without deletions/hallucinations)")
    print("-" * 40)
    print(f"🌟 STRICT ACCURACY        : {strict_acc:.1f}%  (Grammar AND Meaning are perfect)")
    print("="*40)
    