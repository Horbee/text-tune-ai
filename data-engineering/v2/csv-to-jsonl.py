import pandas as pd

pd.read_csv("data/chapter1_corrupted.csv", sep=";").to_json("data/chapter1_corrupted.jsonl", orient="records", lines=True, force_ascii=False)