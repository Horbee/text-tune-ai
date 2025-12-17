import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# --- Configuration ---
MODEL_ID = "unsloth/Ministral-3-8B-Instruct-2512-bnb-4bit" 
NEW_MODEL_NAME = "models/Ministral-3-8B-GEC-v1"
DATA_FILE = "data/ministral-train-formatted.jsonl" 
MAX_SEQ_LENGTH = 8192 # Unsloth handles long context very efficiently

# --- 1. Load Model & Tokenizer via Unsloth ---
print("Loading Unsloth model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_ID,
    max_seq_length = MAX_SEQ_LENGTH,
    dtype = None, # None = Auto detection (will use BF16 on A6000)
    load_in_4bit = True, # 4-bit quantization
)

# --- 2. Fix Tokenizer for Your Data ---
# Since your JSON rows contain "<s>" and "</s>" explicitly:
tokenizer.add_bos_token = False 
tokenizer.add_eos_token = False
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# --- 3. Add LoRA Adapters ---
print("Applying LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = 32,
    lora_dropout = 0, # Unsloth recommends 0 dropout for optimized kernels
    bias = "none",
    use_gradient_checkpointing = "unsloth", # Uses Unsloth's optimized checkpointing
    random_state = 3407,
)

# --- 4. Load Dataset ---
print("Loading dataset...")
dataset = load_dataset("json", data_files=DATA_FILE, split="train")

# --- 5. Training Arguments ---
print("Configuring trainer...")
sft_config = SFTConfig(
    output_dir = "./results",
    dataset_text_field = "text",
    max_length = MAX_SEQ_LENGTH,
    dataset_num_proc = 2,
    packing = False, # Can set to True for speed boost if data allows
    
    # --- Training Parameters ---
    per_device_train_batch_size = 8,
    gradient_accumulation_steps = 2,
    num_train_epochs = 1,
    learning_rate = 2e-4, # Unsloth supports higher LR comfortably
    embedding_learning_rate = 1e-5,
    
    fp16 = not torch.cuda.is_bf16_supported(),
    bf16 = torch.cuda.is_bf16_supported(), # True for A6000
    
    logging_steps = 25,
    save_strategy = "steps",
    save_steps = 100,
    
    warmup_steps = 5,
    optim = "adamw_8bit", # 8-bit optimizer saves even more memory
    weight_decay = 0.01,
    lr_scheduler_type = "linear",
    seed = 3407,
    report_to = "none",
)

# --- 6. Initialize Trainer ---
trainer = SFTTrainer(
    model = model,
    processing_class = tokenizer,
    train_dataset = dataset,
    args = sft_config,
)

# --- 7. Start Training ---
print("Starting training...")
trainer_stats = trainer.train()

# --- 8. Save Model ---
print(f"Saving model to {NEW_MODEL_NAME}...")
# Unsloth allows saving just the adapters (Lora) or the merged model
model.save_pretrained(NEW_MODEL_NAME) # Saves adapters
tokenizer.save_pretrained(NEW_MODEL_NAME)

# If you want to save the merged 16-bit model for vLLM/GGUF later:
# model.save_pretrained_merged(NEW_MODEL_NAME, tokenizer, save_method = "merged_16bit")