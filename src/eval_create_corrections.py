from ollama import chat, ChatResponse
import pandas as pd
import json
import argparse

parser = argparse.ArgumentParser(
            prog='ollama-create-corrections',
            description='Try out a text correction model',
            usage='python ollama-create-corrections.py ministral-3:8b',
        )

parser.add_argument('model_name', type=str, help='Name of the Ollama model to use, e.g., ministral-3:8b')
args = parser.parse_args()

df = pd.read_json("eval.jsonl", lines=True)
safe_model_name = args.model_name.replace(':', '-')
output_file = f'{safe_model_name}-corrected.jsonl'

for index, row in df.iterrows():
    print(f"\nProcessing row: {index}...")
    original_text = row['original']
    user_input = row['corrupted']

    try:
        response: ChatResponse = chat(model=args.model_name, messages=[
                                # Uncomment if you want to use a system prompt
                                {
                                    'role': 'system',
                                    'content': 'Korrigiere die Grammatik im folgenden Text, aber behalte den ursprünglichen Stil und Ton bei. Verleihe dem Text keine formelle Note, wenn er diese nicht hat. Gib **nur** den korrigierten Satz zurück, ohne Anmerkungen. Wenn der Satz korrekt ist, gib ihn unverändert zurück.',
                                },
                                {
                                    'role': 'user',
                                    'content': user_input,
                                },
                        ])
                        
        corrected_text = response["message"]["content"].strip()

        # Append single result to file (much more efficient than rewriting entire file)
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps({'input': user_input, 'model_corrected': corrected_text, 'label': original_text, }, ensure_ascii=False))
            f.write('\n')
        
        print(f"Progress saved: {index + 1}/{len(df)} rows")
                    
    except Exception as e:
        print(f"Error processing row {index}: {e}")