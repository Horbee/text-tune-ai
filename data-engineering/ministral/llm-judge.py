import argparse
import json
import os
import sys
from tqdm import tqdm
import ollama 
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

parser = argparse.ArgumentParser(
            prog='llm-judge.py',
            description='Evaluates the quality of corrupted sentences for German GEC training using an LLM.',
            usage='python llm-judge.py --input <input_path> --correct_column <column_name> --corrupt_column <column_name> --output <output_path>',
        )

parser.add_argument('--input', type=str, default="data/news_data_train-errors.jsonl", help='Path to input JSONL file containing corrupted sentences.')
parser.add_argument('--correct_column', type=str, default="de_correct", help='Name of the column containing clean sentences.')
parser.add_argument('--corrupt_column', type=str, default="de_corrupt", help='Name of the column to store sentences with errors.')
parser.add_argument('--output', type=str, default="data/news_data_train-evaluated.jsonl", help='Path to output JSONL file for sentences with evaluation.')
args = parser.parse_args()


API_KEY = os.getenv("OLLAMA_API_KEY")

client = ollama.Client(
    # host="https://ollama.com",
    # headers={'Authorization': 'Bearer ' + API_KEY}
)

def get_system_instructions():
    return """
        You are a strict Data Quality Judge for a German Grammatical Error Correction (GEC) dataset.
        I will provide you with a CLEAN (correct) sentence and a CORRUPTED (error-filled) sentence derived from it.

        Your task is to determine if the CORRUPTED sentence is a "High Quality" training example.

        CRITERIA FOR "HIGH QUALITY":
        1. unambiguous_error: The corrupted sentence must be clearly ungrammatical or unnatural.
        2. single_intent: A human reader would immediately guess the CLEAN sentence is the intended meaning.
        3. not_valid_alternative: The corrupted sentence must NOT be a valid German sentence with a different meaning (e.g., do not simply change tense "ging" -> "geht" or subject "Er" -> "Sie" if both are valid).
        4. recoverable: It must be possible to reconstruct the Clean sentence without hallucinating new information.

        Input Format:
        Clean: [Sentence]
        Corrupt: [Sentence]

        Output Format (JSON):
        {
        "verdict": "KEEP" or "DISCARD",
        "reason": "Short explanation of why."
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
    """
    
def validate_corruption_pair(clean, corrupt, model="ministral-3:8b"): #model="qwen3-next:80b-cloud" ): #"gemini-3-pro-preview"):
    """
    Asks the LLM if the corruption is valid for GEC training.
    """
    
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
                {"role": "system", "content": get_system_instructions()},
                {"role": "user", "content": prompt}
            ],
            # temperature=0.0,
            # format="json"
        )
        
        result_text = response['message']['content'].strip()
        print(f"LLM Response: {result_text}")

        result_text = result_text.replace("```json", '').replace("```", '')  # Remove code block markers for valid JSON
        return json.loads(result_text)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)



df = pd.read_json(args.input, lines=True)
df_result = pd.read_json(args.output, lines=True)
candidates = list(zip(df[args.correct_column], df[args.corrupt_column]))[len(df_result):50]

print(f"Validating {len(candidates) - len(df_result)} pairs...")

for clean, corrupt in tqdm(candidates[len(df_result):]):
    evaluation = validate_corruption_pair(clean, corrupt)
    
    evaluated_data = { "input": corrupt, "label": clean, "verdict": evaluation["verdict"], "reason": evaluation.get("reason", None) }

    with open(args.output, "a", encoding="utf-8") as f:
        f.write(json.dumps(evaluated_data, ensure_ascii=False) + "\n")
