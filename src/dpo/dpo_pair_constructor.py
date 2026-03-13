import argparse
import pandas as pd
from tqdm import tqdm
from src.prompts import get_inference_prompt_v5
from ollama_similarity_score import get_similarity_score

parser = argparse.ArgumentParser(
            prog='dpo_pair_constructor',
            description='Script for constructing pairs of (model_output, label) for DPO training',
            usage='python dpo_pair_constructor.py --input eval.jsonl',
        )

parser.add_argument('--input', type=str, required=True, help='Path to the input JSONL file, e.g., eval.jsonl')
parser.add_argument('--output', type=str, required=True, help='Path to the output JSONL file, e.g., dpo_pairs.jsonl')
args = parser.parse_args()

df = pd.read_json(args.input, lines=True)
candidates = df.to_dict(orient="records")

dpo_pairs = []
success_count = 0

# adding similarity scores to candidates in batch of 50s
for i in tqdm(range(0, len(candidates), 50), desc="Calculating similarity scores"):
    batch = candidates[i:i+50]
    ground_truths = [item["label"] for item in batch]
    model_outputs = [item["model_corrected"] for item in batch]
    
    scores = get_similarity_score(ground_truths, model_outputs, show_progress=False)
    
    for item, score in zip(batch, scores):
        item["similarity_score"] = score
    

for item in tqdm(candidates, desc="Constructing DPO pairs"):
        input_text = item["input"]
        ground_truth = item["label"]
        model_output = item["model_corrected"]
        score = item["similarity_score"]

        full_prompt = get_inference_prompt_v5(input_text)

        if model_output == ground_truth:
            success_count += 1
            continue
            
        # LAZINESS: The model just copied the input.
        if model_output.strip() == input_text.strip() and ground_truth.strip() != input_text.strip():
            dpo_pairs.append({
                "type": "laziness",
                "prompt": full_prompt, 
                "chosen": ground_truth, 
                "rejected": model_output
            })

        # SIMILARITY: The model output is very similar to the input, but the label is different.
        if score < 0.95:
            dpo_pairs.append({
                "type": f"sim-{score:.2f}",
                "prompt": full_prompt,
                "chosen": ground_truth,
                "rejected": model_output
            })

pd.DataFrame(dpo_pairs).to_json(args.output, orient="records", lines=True, force_ascii=False)
print(f"Total pairs constructed for DPO: {len(dpo_pairs)}")
print(f"Total cases where model output was correct (skipped for DPO): {success_count}")