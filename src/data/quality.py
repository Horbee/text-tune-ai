import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import ollama 
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

class EvalResult(BaseModel):
    reason: str
    verdict: str

SYSTEM_PROMPT = """
You are a strict Data Quality Judge for a German Grammatical Error Correction (GEC) dataset.
I will provide you with a CLEAN (correct) sentence and a CORRUPTED (error-filled) sentence derived from it.

Your task is to determine if the CORRUPTED sentence is a "High Quality" training example.

CRITERIA FOR "HIGH QUALITY":
1. unambiguous_error: The corrupted sentence must be clearly ungrammatical or unnatural.
2. single_intent: A human reader would immediately guess the CLEAN sentence is the intended meaning.
3. not_valid_alternative: The corrupted sentence must NOT be a valid German sentence with a different meaning (e.g., do not simply change tense "ging" -> "geht" or subject "Er" -> "Sie" if both are valid).
4. recoverable: It must be possible to reconstruct the exact Clean sentence. If the Corrupted sentence adds an extra word (like an adjective or adverb), a corrector would naturally try to fix its grammar rather than delete it. These additions make the Clean target unrecoverable.
5. meaning_preserved: The corrupted sentence should not lose crucial information or add new descriptive words/facts that are not in the clean sentence.

Input Format:
Clean: [Sentence]
Corrupt: [Sentence]

Output Format (JSON):
{
    "verdict": "KEEP" or "DISCARD",
    "reason": "Short, one line explanation of why."
}

EXAMPLES:

Clean: "Er ging nach Hause."
Corrupt: "Er geht nach Hause."
Output: {"verdict": "DISCARD", "reason": "Corrupt version is a valid present tense sentence. Tense change is arbitrary."}

Clean: "Das Haus ist groß."
Corrupt: "Das Haus ist gros."
Output: {"verdict": "KEEP", "reason": "Clear spelling error, recoverable."}

Clean: "Wir essen Brot."
Corrupt: "Wir essen."
Output: {"verdict": "DISCARD", "reason": "Corrupt version is valid (intransitive usage). Context is lost."}

Clean: "Die Frau liest das Buch."
Corrupt: "Die junge Frau liest das Buch."
Output: {"verdict": "DISCARD", "reason": "Corrupted version adds the adjective 'junge'. A GEC model would just fix grammar, not delete the word, failing to match the clean target."}
"""
    
def validate_corruption_pair(clean: str, corrupt: str, client: ollama.Client, model: str = "gpt-oss:120b-cloud") -> EvalResult:
    prompt = f"""
    Analyze this pair based on the system instructions.
    
    Clean: "{clean}"
    Corrupt: "{corrupt}"
    
    Return ONLY JSON.
    """
    
    try:
        response = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            format=EvalResult.model_json_schema(),
            options={'temperature': 0.0}, # Temperature 0 for maximum consistency
        )
        
        result_str = response['message']['content']
        return json.loads(result_str)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                prog='quality.py',
                description='Evaluates the quality of corrupted sentences for German GEC training using an LLM.',
                usage='python quality.py --input <input_path> --correct_column <column_name> --corrupt_column <column_name> --output <output_path>',
            )

    parser.add_argument('--input', type=str, required=True, help='Path to input JSONL file containing corrupted sentences.')
    parser.add_argument('--output', type=str, required=True, help='Path to output JSONL file for sentences with evaluation.')
    parser.add_argument('--correct_column', type=str, default="original", help='Name of the column containing clean sentences.')
    parser.add_argument('--corrupt_column', type=str, default="corrupted", help='Name of the column to store sentences with errors.')
    parser.add_argument('--from_index', type=int, default=0, help='Starting index for processing sentences.')
    parser.add_argument('--model', type=str, default="gpt-oss:120b-cloud", help='Model to use for evaluation.')
    args = parser.parse_args()



    client = ollama.Client(
        host="https://ollama.com",
        headers={'Authorization': 'Bearer ' + os.getenv("OLLAMA_API_KEY", "")}
    )


    print(f"Reading input from {args.input}...")

    df = pd.read_json(args.input, lines=True)
    candidates = list(zip(df[args.correct_column], df[args.corrupt_column]))[args.from_index:]

    BATCH_SIZE = 3

    def process_pair(pair):
        clean, corrupt = pair
        evaluation = validate_corruption_pair(clean, corrupt, client, args.model)
        return { "input": corrupt, "label": clean, "verdict": evaluation["verdict"], "reason": evaluation["reason"] }

    with tqdm(total=len(candidates), desc="Evaluating pairs") as pbar:
        for i in range(0, len(candidates), BATCH_SIZE):
            batch = candidates[i:i + BATCH_SIZE]
            with ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
                futures = {executor.submit(process_pair, pair): pair for pair in batch}
                for future in as_completed(futures):
                    evaluated_data = future.result()
                    with open(args.output, "a", encoding="utf-8") as f:
                        f.write(json.dumps(evaluated_data, ensure_ascii=False) + "\n")
                    pbar.update(1)
