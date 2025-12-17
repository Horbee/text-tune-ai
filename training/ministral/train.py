import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    Mistral3ForConditionalGeneration,
    AutoConfig,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# --- Configuration ---
MODEL_ID = "mistralai/Ministral-3-8B-Instruct-2512" 
NEW_MODEL_NAME = "models/Ministral-3-8B-GEC-v1"
DATA_FILE = "data/ministral-train-formatted.jsonl" 

# --- 1. Load Config & Fix FP8 Conflict ---
print("Loading configuration...")
config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)

# FIX: Remove the existing FP8 quantization config to avoid conflict with BitsAndBytes
if hasattr(config, "quantization_config"):
    del config.quantization_config

# --- 2. Quantization Configuration (4-bit) ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, # Optimal for Ampere (A6000)
    bnb_4bit_use_double_quant=True,
)

# --- 3. Load Model ---
print("Loading model...")
model = Mistral3ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    config=config,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    # attn_implementation="flash_attention_2" # A6000 supports Flash Attention 2
)

# Enable gradient checkpointing to save memory (optional on 48GB, but good for stability)
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# --- 4. Load Tokenizer (With Fixes) ---
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Fix for FP16/BF16 training stability

# --- 5. LoRA Configuration ---
peft_config = LoraConfig(
    r=16,       # Rank
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj"
    ],
)

# --- 6. Training Arguments (Optimized for A6000) ---
sft_config = SFTConfig(
    output_dir="./results",
    max_length=8192,             
    dataset_text_field="text",       
    packing=False,                   
    # --- Standard Training Parameters ---
    num_train_epochs=1,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    logging_steps=25,
    save_strategy="steps",
    save_steps=100,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=True,                       # Keep True for A6000
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="cosine",
    report_to="none",
    
    # --- Memory Optimization ---
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False}, # specific fix for some versions
)

print("Loading dataset...")
dataset = load_dataset("json", data_files=DATA_FILE, split="train")

# --- 7. Initialize Trainer ---
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    processing_class=tokenizer,
    args=sft_config,
)

# --- 8. Start Training ---
print("Starting training...")
trainer.train()

# --- 9. Save Model ---
print("Saving model...")
trainer.model.save_pretrained(NEW_MODEL_NAME)
tokenizer.save_pretrained(NEW_MODEL_NAME)
print(f"Model saved to {NEW_MODEL_NAME}")