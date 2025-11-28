import os
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse

parser = argparse.ArgumentParser(
            prog='data_splitter',
            description='Splits dataset for training and evaluation',
            usage='python data_splitter.py ./data/chapter1-eval-v4.jsonl',
        )

parser.add_argument('data_path')
parser.add_argument('--seed', type=int, nargs='?', default=42, help='Random seed for splitting the dataset')
args = parser.parse_args()

print(args)

df = pd.read_json(
    os.path.join(os.path.dirname(__file__), args.data_path),
    lines=True
)

print(f"Total samples: {len(df)}")

train_df, val_df = train_test_split(
    df, test_size=0.02, random_state=args.seed
)  # 98/2 split

print(f"Train: {len(train_df)}, Val: {len(val_df)}")

output_dir = os.path.dirname(args.data_path)
filename = os.path.basename(args.data_path)
extension = filename.split('.')[-1]
name_only = filename.replace(f'.{extension}', '')
output_train_path = os.path.join(output_dir, f'{name_only}-train.{extension}')
output_val_path = os.path.join(output_dir, f'{name_only}-eval.{extension}')

train_df.to_json(output_train_path, lines=True, orient='records', force_ascii=False)
val_df.to_json(output_val_path, lines=True, orient='records', force_ascii=False)

print("Datasets saved...")