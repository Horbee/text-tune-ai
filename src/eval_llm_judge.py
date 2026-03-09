import json
import os
from ollama import Client
from pydantic import BaseModel
from tqdm import tqdm
from dotenv import load_dotenv
import pandas as pd
import argparse

load_dotenv()  # Load environment variables from .env file

parser = argparse.ArgumentParser(
            prog='evaluate-gec-ollama',
            description='Try out a text correction model',
            usage='python evaluate_gec_ollama.py <filename>',
        )

parser.add_argument('correction_file', type=str, help='Path to the correction file, e.g., ministral-3-14b-corrected.jsonl')
args = parser.parse_args()

OUTPUT_FILE = f"{args.correction_file}-results.jsonl"

JUDGE_MODEL = "gpt-oss:120b-cloud"

class EvalResult(BaseModel):
    is_grammatically_correct: bool
    meaning_preserved: bool
    reason: str

client = Client(
        host="https://ollama.com",
        headers={'Authorization': 'Bearer ' + os.environ.get('OLLAMA_API_KEY')}
    )

# Your evaluation dataset:
# "original" is the broken sentence you fed to your fine-tuned model.
# "model_output" is what your fine-tuned model actually generated.
eval_data = pd.read_json(args.correction_file, lines=True)
dataset_dict = eval_data.to_dict(orient="records")  # Convert to list of dicts for easier processing

def evaluate_correction(original, model_output):
    prompt = f"""
You are an expert German linguist evaluating a Grammatical Error Correction (GEC) system.

Evaluate the MODEL OUTPUT on two criteria:
1. is_grammatically_correct: Is the MODEL OUTPUT completely free of grammatical, spelling, case (Kasus), and punctuation errors? (true/false)
2. meaning_preserved: Does the MODEL OUTPUT preserve the exact intended meaning of the ORIGINAL sentence without deleting crucial information or hallucinating new facts? (true/false)

ORIGINAL: "{original}"
MODEL OUTPUT: "{model_output}"

Return ONLY valid JSON in this exact format, with no markdown formatting or extra text:
{{
  "is_grammatically_correct": true,
  "meaning_preserved": true,
  "reason": "short explanation"
}}
"""

    try:
        # We use format='json' to force Ollama to return a valid JSON object
        response = client.chat(
            model=JUDGE_MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.0}, # Temperature 0 for maximum consistency
            format=EvalResult.model_json_schema(),
        )
        
        result_str = response['message']['content']
        return json.loads(result_str)
        
    except Exception as e:
        print(f"\nError parsing JSON for input: '{original}'. Error: {e}")
        # Default to False if the model fails to format correctly
        return {
            "is_grammatically_correct": False, 
            "meaning_preserved": False, 
            "reason": "Parsing Error"
        }


if __name__ == "__main__":
    print(f"Loading {JUDGE_MODEL} for evaluation...\n")
    
    results = []
    strict_correct_count = 0
    grammar_correct_count = 0
    meaning_preserved_count = 0

    for item in tqdm(dataset_dict, desc="Judging Output", total=len(dataset_dict)):
        original = item["label"]
        output = item["model_corrected"]
        
        # Ask the LLM Judge
        judgment = evaluate_correction(original, output)
        
        # Track individual metrics
        is_grammar_ok = judgment.get("is_grammatically_correct", False)
        is_meaning_ok = judgment.get("meaning_preserved", False)
        
        if is_grammar_ok:
            grammar_correct_count += 1
        if is_meaning_ok:
            meaning_preserved_count += 1
            
        # STRICT ACCURACY: Both must be True!
        if is_grammar_ok and is_meaning_ok:
            strict_correct_count += 1
            
        # Save to log
        item.update(judgment)
        results.append(item)

        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

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
    