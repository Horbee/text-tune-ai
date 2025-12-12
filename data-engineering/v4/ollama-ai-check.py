import pandas as pd
import ollama

# Read the data
df = pd.read_json('data/news_data_text_30000.jsonl', lines=True)

# Process in chunks of 1000
chunk_size = 1000
all_results = []  # Store both IDs and explanations
max = 1000#len(df)

for start_idx in range(0, max, chunk_size):
    end_idx = min(start_idx + chunk_size, max)
    df_chunk = df[start_idx:end_idx]
    
    print(f"\nProcessing rows {start_idx} to {end_idx}...")
    
    # Prepare batch
    batch_data = []
    for idx, row in df_chunk.iterrows():
        text_id = row.get('id', idx)
        text_content = row.get('text', row.get('content', ''))
        batch_data.append(f"ID: {text_id} | Text: {text_content}")
    
    batch_prompt = "\n\n".join(batch_data)
    
    try:
        response = ollama.chat(model='mistral-large-3:675b-cloud', messages=[
            {
                'role': 'system',
                'content': '''Check the following texts and identify which ones contain serious grammatically incorrect sentences.
                Misformed sentences, incomplete sentences, fragments or misformed sentences should be considered grammatically incorrect. 
                For each grammatically incorrect text, provide:
                1. The ID
                2. A brief explanation of the grammatical error
                
                Format your response as:
                ID: [id] | Explanation: [brief explanation]
                
                Use one line per incorrect text. If all texts are correct, return: NONE''',
            },
            {
                'role': 'user',
                'content': batch_prompt,
            }
        ])
        
        result = response['message']['content'].strip()
        print(f"Model response:\n{result}\n")
        
        if result.upper() != 'NONE':
            # Parse each line
            lines = result.split('\n')
            for line in lines:
                if 'ID:' in line and 'Explanation:' in line:
                    try:
                        # Extract ID and explanation
                        parts = line.split('|')
                        id_part = parts[0].split('ID:')[1].strip()
                        explanation_part = parts[1].split('Explanation:')[1].strip()
                        all_results.append({'id': id_part, 'explanation': explanation_part})
                    except Exception as parse_error:
                        print(f"Could not parse line: {line}")
            
            print(f"Found {len([l for l in lines if 'ID:' in l])} incorrect texts")
        else:
            print("No incorrect texts found")
            
    except Exception as e:
        print(f"Error processing batch: {e}")

# Write results to CSV
if all_results:
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('data/incorrect_grammar_ids.csv', index=False)
    print(f"\nTotal incorrect sentences found: {len(all_results)}")
    print(f"Results written to 'data/incorrect_grammar_ids.csv'")
else:
    print("\nNo incorrect sentences found in any batch")
