from transformers import pipeline
import argparse

parser = argparse.ArgumentParser(
                    prog='correction-test',
                    description='Test a text correction model',
                    usage='python pipeline-correction.py ./tf-results/checkpoint-23349 "ich finde mein Brille nicht" -p "grammar: "',
                    # usage='python pipeline-correction.py ./gec_german_mt5_model/checkpoint-185 "hast du den Buch gelesen?" -p "korrigiere: "',
                    # usage='python pipeline-correction.py ./final_gec_mt5_model "hast du den Buch gelesen?" -p "korrigiere: "',
        )

parser.add_argument('model_name')           # positional argument
parser.add_argument('corrupted_text')           # positional argument
parser.add_argument('-p', '--prefix', default="korrigiere: ")      # option that takes a value

args = parser.parse_args()

pipe = pipeline(
    task="text2text-generation",
    # model="google-t5/t5-base",
    # model="./tf-results/checkpoint-23349",
    model=args.model_name,
    # dtype=torch.float16,
    # device=0
)

out = pipe(f"{args.prefix}{args.corrupted_text}")

print(out)
