import pandas as pd


df = pd.read_json('../mt5/data/news-data-v5-train.jsonl', lines=True)
new_df = pd.DataFrame(columns=['text'])

counter = 0

for index, row in df.iterrows():
    if counter >= 10100:
        break

    sentence = row['de_correct']
    if len(sentence.split()) < 5:
        continue

    new_df.at[counter, 'text'] = sentence
    counter += 1

new_df.to_json('ministral-data.jsonl', orient='records', force_ascii=False, lines=True)