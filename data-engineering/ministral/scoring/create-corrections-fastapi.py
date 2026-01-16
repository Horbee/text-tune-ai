import pandas as pd
import json
import requests


df = pd.read_json("data/chapter1-eval-v4.jsonl", lines=True)
output_file = 'ministral-3:8b-GEC-q8-v2-corrected.jsonl'

for index, row in df.iterrows():
    print(f"\nProcessing row: {index}...")
    original_text = row['de_correct']
    user_input = row['de_corrupted']

    try:
        response = requests.post(
            'http://localhost:3000/api/gec',
            # 'http://localhost:8000/gec',
            json={'text': user_input}
        )

        corrected_text = response.json().get('corrected', '').strip()

        # Append single result to file (much more efficient than rewriting entire file)
        with open(output_file, 'a', encoding='utf-8') as f:
            json.dump({'original_text': original_text, 'corrected_text': corrected_text}, f, ensure_ascii=False)
            f.write('\n')
        
        print(f"Progress saved: {index + 1}/{len(df)} rows")
                    
    except Exception as e:
        print(f"Error processing row {index}: {e}")