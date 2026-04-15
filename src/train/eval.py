from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "./models/Ministral-3-14B-Instruct-2512-GEC-v6", # Point directly to the ADAPTER folder
    max_seq_length = 8192,          # Use the same as your training
    load_in_4bit = True,            # Set to True if you trained with 4-bit
    dtype = None,                   # Auto-detect (Float16/Bfloat16)
    # device_map="cpu"
)