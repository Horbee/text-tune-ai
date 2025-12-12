import pandas as pd
import ollama

df = pd.read_csv('quelle_sentences.csv')

all_results = []  

for idx in range(10, 1000):
    print(f"\nProcessing row: {idx}...")

    text_id = df.iloc[idx].get('id')
    text_content = df.iloc[idx].get('text')

    try:
        response = ollama.chat(model='llama3.2', messages=[
            {
            "role": "system",
            "content": '''You will receive a sentence that may contain grammatical errors, structural problems, or unwanted annotations starting with 'Quelle'. 
                         Your task is to:
                         
                         1. Remove all 'Quelle' annotations, including everything directly following them until the next sentence fragment.
                         2. Remove any structurally broken or duplicated parts caused by these annotations.
                         3. Fix the grammar and structure of the remaining text.
                         4. Return **only** the corrected sentence, with no explanations or additional comments.
                         
                         Example:
                         Incorrect: \"Quelle: picture alliance / Uwe Gerig/picture alliance 18 von 19 Die tschechische Bildhauerin Marie Uchytilova hat im Andenken an die 42 Mädchen und 40 Jungen, die von den Deutschen im Juni 1942 ermordet wurden ... Quelle: picture alliance / Uwe Gerig/picture alliance 19 von 19 ... die Bronzestatuengruppe Denkmal für die Kinderopfer des Krieges geschaffen.”
                         Correct: \"Die tschechische Bildhauerin Marie Uchytilova hat im Andenken an die 42 Mädchen und 40 Jungen, die von den Deutschen im Juni 1942 ermordet wurden, die Bronzestatuengruppe Denkmal für die Kinderopfer des Krieges geschaffen.”'''
            },
            {
                'role': 'user',
                'content': text_content,
            }
        ])
        
        result = response['message']['content'].strip()
        print(f"Model response:\n{result}\n")
        
        all_results.append({
            'id': text_id,
            'original_text': text_content,
            'corrected_text': result
        })
                    
    except Exception as e:
        print(f"Error processing row {idx}: {e}")


results_df = pd.DataFrame(all_results)
results_df.to_json('data/removed_quelle_sentences1.jsonl', lines=True, index=False, force_ascii=False, orient='records')
print(f"Results written to 'data/removed_quelle_sentences1.jsonl'")
