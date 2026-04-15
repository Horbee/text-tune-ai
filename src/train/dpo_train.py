# ==========================================
# CRITICAL UNSLOTH STEP: Patch TRL
# ==========================================
# You MUST import and run PatchDPOTrainer before doing anything else with TRL.
# This injects Unsloth's highly optimized math into the standard HuggingFace trainer.
from unsloth import FastLanguageModel, PatchDPOTrainer

PatchDPOTrainer()

import json
import argparse
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer


# ==========================================
# HOTFIX: Prevent TRL from hallucinating a vision model
# ==========================================
_original_prepare_dataset = DPOTrainer._prepare_dataset


def patched_prepare_dataset(self, *args, **kwargs):
    self.is_vision_model = False  # Force it back to text-only mode
    return _original_prepare_dataset(self, *args, **kwargs)


DPOTrainer._prepare_dataset = patched_prepare_dataset
# ==========================================


def main(args):
    # ==========================================
    # 1. HYPERPARAMETERS & CONFIGURATION
    # ==========================================
    MODEL_ID = args.model
    DATASET_PATH = args.dataset
    OUTPUT_DIR = args.output_dir

    # Gentle LoRA Settings
    LORA_R = 8
    LORA_ALPHA = 16
    TARGET_MODULES = ["q_proj", "v_proj"]

    # DPO & Training Settings
    MAX_SEQ_LENGTH = 512  # Unsloth handles RoPE scaling automatically
    BETA = 1.0
    LEARNING_RATE = 2e-6
    EPOCHS = 1
    BATCH_SIZE = 4
    GRAD_ACCUMULATION = 4

    # ==========================================
    # 2. MODEL & TOKENIZER (UNSLOTH WAY)
    # ==========================================
    print(f"Loading Unsloth model: {MODEL_ID}...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # Unsloth auto-detects fp16 or bf16 based on your GPU
        load_in_4bit=True,  # Use 4-bit quantization to save massive VRAM
    )

    # ==========================================
    # 3. LoRA CONFIGURATION (UNSLOTH WAY)
    # ==========================================
    # Unsloth has its own optimized get_peft_model function
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,  # CRITICAL: Unsloth explicitly requires dropout to be 0 for speed
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth's custom checkpointing saves ~30% VRAM
        random_state=3407,
    )

    # ==========================================
    # 4. DATA PREPARATION
    # ==========================================
    print(f"Loading dataset from {DATASET_PATH}...")
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

    # Define a function to dynamically append the correct EOS token
    def append_eos_token(row):
        # 1. Build a real Python dictionary using your clean text
        chosen_dict = {"corrected_text": row["chosen"]}
        rejected_dict = {"corrected_text": row["rejected"]}

        # 2. Safely convert it to a JSON string (ensure_ascii=False keeps German umlauts intact)
        chosen_json_string = json.dumps(chosen_dict, ensure_ascii=False)
        rejected_json_string = json.dumps(rejected_dict, ensure_ascii=False)

        # 3. Append the EOS token
        row["chosen"] = chosen_json_string + tokenizer.eos_token
        row["rejected"] = rejected_json_string + tokenizer.eos_token
        return row

    # Apply the function to your dataset
    print(f"Appending EOS token ({tokenizer.eos_token}) to responses...")
    dataset = dataset.map(append_eos_token)

    # Optional: Split for evaluation
    dataset = dataset.train_test_split(test_size=0.05)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    # ==========================================
    # 5. DPO TRAINER SETUP
    # ==========================================
    training_args = DPOConfig(
        output_dir=OUTPUT_DIR,
        beta=BETA,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        num_train_epochs=EPOCHS,
        max_length=MAX_SEQ_LENGTH,
        max_prompt_length=MAX_SEQ_LENGTH // 2,
        # Unsloth runs exceptionally well with 8-bit AdamW
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_steps=5,
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="steps" if len(eval_dataset) > 0 else "no",
        eval_steps=20,
        remove_unused_columns=False,
    )

    # --- HOTFIX FOR TRL BUG ---
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}
    # --------------------------

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # Unsloth/TRL handles the reference model natively in memory
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer.tokenizer
        if hasattr(tokenizer, "tokenizer")
        else tokenizer,
    )

    # ==========================================
    # 6. TRAIN & SAVE
    # ==========================================
    print("Starting Unsloth DPO training...")
    trainer.train()

    print("Saving LoRA adapters...")
    # Use Unsloth's native save function to ensure compatibility
    model.save_pretrained(f"{OUTPUT_DIR}")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}")

    print("Training complete! Adapters saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DPO Training with Unsloth")
    parser.add_argument(
        "--model",
        type=str,
        default="unsloth/Ministral-3-3B-Instruct-2512",
        help="HuggingFace Model ID to train",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/processed/dpo_dataset.jsonl",
        help="Path to the JSONL dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./gec-3b-dpo-unsloth",
        help="Output directory for the trained adapters",
    )

    args = parser.parse_args()
    main(args)
