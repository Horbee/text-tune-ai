import json
import argparse
from transformers import AutoTokenizer
from src.prompts import get_train_prompt_v5


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Corrupt German sentences for GEC dataset.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the input JSONL file.")
    args = parser.parse_args()
        
    # 1. Load the exact tokenizer you are using for training
    tokenizer = AutoTokenizer.from_pretrained("unsloth/Ministral-3-3B-Instruct-2512")

    max_tokens = 0
    longest_example = ""

    # 3. Scan the dataset
    print("Scanning dataset to find max token length...")
    with open(args.dataset, "r", encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            corrupted = row["corrupted"]
            original = row["original"]
            
            # Format the full sequence exactly as it appears in training
            full_sequence = get_train_prompt_v5(corrupted, original)
            
            # Tokenize and count
            token_count = len(tokenizer(full_sequence)["input_ids"])
            
            if token_count > max_tokens:
                max_tokens = token_count
                longest_example = full_sequence

    print(f"\n✅ Scan Complete!")
    print(f"The absolute longest sequence in your dataset is: **{max_tokens} tokens**")
    print(f"Example of the longest sequence:\n{longest_example}\n")

    # Add a small safety buffer (e.g., rounding up to the nearest power of 2, or just adding 64)
    recommended_length = ((max_tokens // 64) + 1) * 64 
    print(f"Recommended MAX_SEQ_LENGTH: **{recommended_length}**")