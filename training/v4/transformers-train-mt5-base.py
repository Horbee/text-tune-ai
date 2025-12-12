#!/usr/bin/env python3
"""
Optimized mT5 training script for German Grammar Error Correction (GEC).
Includes memory optimization, mixed precision, gradient accumulation, and better monitoring.
"""

from datasets import load_dataset
from transformers import (
    T5Tokenizer,
    MT5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import evaluate
import numpy as np
from peft import LoraConfig, get_peft_model, TaskType
from accelerate import Accelerator
import os
import torch
from accelerate_logging import MyLogger


# --- Accelerate & Logging Setup ---
accelerator = Accelerator()
logger = MyLogger(accelerator)

# --- Configuration ---
MODEL_CHECKPOINT = "google/mt5-base"
MAX_LENGTH = 128
BATCH_SIZE = 16  # Increased for efficiency
GRADIENT_ACCUMULATION_STEPS = 2  # Effective batch = 32
LEARNING_RATE = 5e-4  # Conservative for stability
WARMUP_RATIO = 0.15  # Extended warmup
NUM_EPOCHS = 5  # Sufficient for LoRA; early stopping will handle it
SEED = 42

# Check GPU availability
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if accelerator.is_main_process:
    logger.info(f"Using device: {DEVICE}")
    if DEVICE == "cuda":
        try:
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        except Exception:
            # In multi-process setups, get_device_name may race; ignore safely
            pass

# --- 1. Load & Split Dataset ---
logger.info("Loading dataset...")
train_path = os.path.join(os.path.dirname(__file__), "data/news-data-v5-train.jsonl")
val_path = os.path.join(os.path.dirname(__file__), "data/news-data-v5-eval.jsonl")
dataset = load_dataset('json', data_files={'train': train_path, 'validation': val_path})
logger.info(f"Train samples: {len(dataset['train'])}, Val samples: {len(dataset['validation'])}")

# --- 2. Tokenizer & Preprocessing ---
logger.info("Loading tokenizer...")
tokenizer = T5Tokenizer.from_pretrained(
    MODEL_CHECKPOINT, use_fast=False, legacy=False, model_max_length=MAX_LENGTH
)

def preprocess_function(examples):
    inputs = examples["de_corrupted"]
    targets = examples["de_correct"]

    model_inputs = tokenizer( # ← treated as encoder input, not decoder target
        inputs, max_length=MAX_LENGTH, truncation=True, padding="max_length", return_tensors=None
    )

    labels = tokenizer(
        text_target=targets, # ← treated as decoder target
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length",
        return_tensors=None
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

logger.info("Tokenizing dataset...")
tokenized_datasets = dataset.map(
    preprocess_function, batched=True, batch_size=32, remove_columns=dataset["train"].column_names
)

# --- 3. Metrics (BLEU) ---
metric = evaluate.load("sacrebleu")
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    # Clip to valid vocabulary range
    vocab_size = len(tokenizer)
    preds = np.where(preds < vocab_size, preds, tokenizer.pad_token_id)
    preds = np.where(preds >= 0, preds, tokenizer.pad_token_id)

    # Replace -100 labels
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # Decode
    # decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # decoded_preds = [pred.strip() for pred in decoded_preds]
    # decoded_labels = [label.strip() for label in decoded_labels]
    decoded_preds = [
        tokenizer.decode(ids, skip_special_tokens=True).strip() for ids in preds
    ]
    decoded_labels = [
        tokenizer.decode(ids, skip_special_tokens=True).strip() for ids in labels
    ]

    # Log samples
    # logger.info("\n--- Sample Predictions ---")
    # for pred, label in zip(decoded_preds[:3], decoded_labels[:3]):
    #     logger.info(f"Pred:  {pred}")
    #     logger.info(f"Label: {label}\n")

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": result["score"]}


# --- 4. Model with LoRA ---
logger.info("Loading model...")
model = MT5ForConditionalGeneration.from_pretrained(MODEL_CHECKPOINT)

# LoRA Configuration
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=8,  # Lower rank for stability
    lora_alpha=16,  # Match 2x rank
    lora_dropout=0.1,
    target_modules=["q", "v"],  # Attention query and value layers
    bias="none",
)

model = get_peft_model(model, peft_config)
if accelerator.is_main_process:
    model.print_trainable_parameters()

# --- 5. Training Arguments ---
training_args = Seq2SeqTrainingArguments(
    output_dir="./checkpoints/gec_german_mt5_optimized",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE * 4,  # Can use larger for eval
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    warmup_ratio=WARMUP_RATIO,
    weight_decay=0.01,
    max_grad_norm=0.5,  # Stricter clipping for stability
    num_train_epochs=NUM_EPOCHS,
    logging_steps=50,
    logging_dir="./logs",
    save_total_limit=5,
    load_best_model_at_end=True,
    metric_for_best_model="bleu",
    greater_is_better=True,
    predict_with_generate=True,
    generation_max_length=MAX_LENGTH,
    generation_num_beams=1,  # Lightweight beam search
    bf16=(DEVICE == "cuda" and torch.cuda.is_bf16_supported()),  # Use bf16 if available
    fp16=False,
    optim="adamw_torch",
    seed=SEED,
    dataloader_num_workers=0,  # Parallel data loading
    dataloader_pin_memory=False,
    push_to_hub=False,
    report_to=["none"],  # Avoid multi-process logging backends unless configured
    ddp_find_unused_parameters=False,  # Safer defaults with accelerate
)

# --- 6. Data Collator & Trainer ---
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding="max_length")

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

# --- 7. Train ---
logger.info("Starting training...")
trainer.train()
# trainer.evaluate()  # For testing purposes, run evaluation only

# --- 8. Save Model ---
output_dir = os.path.join(os.path.dirname(__file__), "models/gec_german_mt5_base")
if trainer.is_world_process_zero():
    os.makedirs(output_dir, exist_ok=True)
    
trainer.save_model(output_dir)
logger.info(f"Model and Tokenizer saved to {output_dir}")
