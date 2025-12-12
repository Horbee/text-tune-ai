import pandas as pd
from tqdm import tqdm
import json
import  os


with open(os.path.join(os.path.dirname(__file__), 'data', 'removed_quelle_sentences4.json')) as f:
    cleaned_df = json.load(f)

df = pd.read_json(os.path.join(os.path.dirname(__file__), 'data', 'news_data_text_30000_cleaned.jsonl'), lines=True)

changed_sentences = 0

for row in tqdm(cleaned_df, total=len(cleaned_df)):
    id = row.get('id')
    corrected_text = row.get('corrected_text')

    df.loc[df['id'] == id, 'text'] = corrected_text
    changed_sentences += 1

df.to_json(os.path.join(os.path.dirname(__file__), 'data', 'news_data_text_30000_cleaned.jsonl'), lines=True, force_ascii=False, orient='records')

print(f"Number of changed sentences: {changed_sentences}")