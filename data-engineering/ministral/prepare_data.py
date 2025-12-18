import json
import os

# --- Configuration ---
INPUT_FILE = "data/ministral-train.jsonl"      # Your original file
OUTPUT_FILE = "data/ministral-train-formatted-v2.jsonl"    # The file to feed into the training script
SYSTEM_PROMPT = "Korrigiere die Grammatik im folgenden Satz auf Standarddeutsch. Gib **nur** den korrigierten Satz zurück, ohne Anmerkungen."

def prepare_dataset():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Could not find {INPUT_FILE}")
        return

    print(f"Processing {INPUT_FILE}...")
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as infile, \
         open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        
        count = 0
        skipped = 0
        
        for line in infile:
            if not line.strip():
                continue
                
            try:
                row = json.loads(line)
                
                # Check if columns exist
                if 'input' not in row or 'label' not in row:
                    skipped += 1
                    continue
                
                bad_sentence = row['input'].strip()
                good_sentence = row['label'].strip()
                
                # --- The Magic Format (Mistral V3 / Tekken) ---
                # 1. <s> : Beginning of Sentence (BOS)
                # 2. [INST] ... [/INST] : The Instruction block
                # 3. </s> : End of Sentence (EOS) - Crucial for the model to stop generating
                
                formatted_text = (
                    f"<s>[INST] {SYSTEM_PROMPT}\n\n"
                    f"{bad_sentence} [/INST] "
                    f"{good_sentence}</s>"
                )
                
                # Write to new file with 'text' column
                new_row = {"text": formatted_text}
                outfile.write(json.dumps(new_row, ensure_ascii=False) + "\n")
                count += 1
                
            except json.JSONDecodeError:
                skipped += 1

    print(f"Done! Processed {count} rows.")
    if skipped > 0:
        print(f"Skipped {skipped} invalid rows.")
    print(f"Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    prepare_dataset()