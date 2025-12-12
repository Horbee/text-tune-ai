import argparse
import pandas as pd

parser = argparse.ArgumentParser(
            prog='cleaner',
            usage='python cleaner.py --drop_nas',
        )

parser.add_argument('-f', '--function-list', nargs='+', default=[])
args = parser.parse_args()


df = pd.read_json('data/news_data_text_30000.jsonl', lines=True)
initial_row_count = len(df)

def drop_nas():
    global df
    df.dropna(subset=['text'], inplace=True)
    df = df[df['text'].str.strip() != '']


def strip_text():
    global df
    df['text'] = df['text'].str.strip()
    
    # Strip repetitive spaces
    df['text'] = df['text'].str.replace(r'\s+', ' ', regex=True)


def remove_duplicates():
    global df
    df.drop_duplicates(subset=['text'], inplace=True)


functions = {
    'drop_nas': drop_nas,
    'remove_duplicates': remove_duplicates,
    'strip_text': strip_text,
}

for func_name in args.function_list:
    if func_name in functions:
        print(f"Executing function: {func_name}")
        functions[func_name]()
    else:
        print(f"Function '{func_name}' not found.")


final_row_count = len(df)
print(f"Initial row count: {initial_row_count}; Final row count: {final_row_count}")
print(f"Rows removed: {initial_row_count - final_row_count}")
df.to_json('data/news_data_text_30000_cleaned.jsonl', lines=True, force_ascii=False, orient='records')