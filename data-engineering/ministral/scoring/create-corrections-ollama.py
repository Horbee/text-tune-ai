from ollama import chat
from ollama import ChatResponse
import pandas as pd
import json

import argparse

parser = argparse.ArgumentParser(
            prog='create-corrections-ollama',
            description='Try out a text correction model',
            usage='python create-corrections-ollama.py ministral-3:8b',
        )

parser.add_argument('model_name', type=str, help='Name of the Ollama model to use, e.g., ministral-3:8b')
args = parser.parse_args()

df = pd.read_json("data/chapter1-eval-v4.jsonl", lines=True)
output_file = f'{args.model_name}-corrected.jsonl'

for index, row in df.iterrows():
    print(f"\nProcessing row: {index}...")
    original_text = row['de_correct']
    user_input = row['de_corrupted']

    try:
        response: ChatResponse = chat(model=args.model_name, messages=[
                                # Uncomment if you want to use a system prompt
                                # {
                                #     'role': 'system',
                                #     'content': 'Korrigiere die Grammatik im folgenden Text, aber behalte den ursprünglichen Stil und Ton bei. Verleihe dem Text keine formelle Note, wenn er diese nicht hat. Gib **nur** den korrigierten Satz zurück, ohne Anmerkungen. Wenn der Satz korrekt ist, gib ihn unverändert zurück.',
                                # },
                                {
                                    'role': 'user',
                                    'content': user_input,
                                },
                        ])
                        
        corrected_text = response["message"]["content"].strip()

        # Append single result to file (much more efficient than rewriting entire file)
        with open(output_file, 'a', encoding='utf-8') as f:
            json.dump({'original_text': original_text, 'corrected_text': corrected_text}, f, ensure_ascii=False)
            f.write('\n')
        
        print(f"Progress saved: {index + 1}/{len(df)} rows")
                    
    except Exception as e:
        print(f"Error processing row {index}: {e}")