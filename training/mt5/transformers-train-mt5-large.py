#!/usr/bin/env python3
"""
Optimized mT5 training script for German Grammar Error Correction (GEC).
Optimized for 2x NVIDIA A6000 (48GB each) with DDP.
Run with: accelerate launch --multi_gpu --num_processes=2 transformers-train-mt5-large.py
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
MODEL_CHECKPOINT = "google/mt5-large"
MAX_LENGTH = 128
BATCH_SIZE = 12  # A6000 has 48GB VRAM - conservative for gradient checkpointing
GRADIENT_ACCUMULATION_STEPS = 3  # Effective batch = 12 * 2 GPUs * 3 = 72
LEARNING_RATE = 2e-4  # Slightly lower for large model stability
WARMUP_RATIO = 0.1  # 10% warmup
NUM_EPOCHS = 3  # LoRA converges quickly; early stopping will handle it
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
dataset = load_dataset('json', data_files={'train': train_path, 'validation': val_path}) # keep_in_memory=True)
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
    preprocess_function, batched=True, batch_size=1000, remove_columns=dataset["train"].column_names,
    # keep_in_memory=True,  # Keep tokenized data in RAM
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

    decoded_preds = [
        tokenizer.decode(ids, skip_special_tokens=True).strip() for ids in preds
    ]
    decoded_labels = [
        tokenizer.decode(ids, skip_special_tokens=True).strip() for ids in labels
    ]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": result["score"]}


# --- 4. Model with LoRA ---
logger.info("Loading model...")
model = MT5ForConditionalGeneration.from_pretrained(MODEL_CHECKPOINT)

# Enable gradient checkpointing - essential for A6000 48GB with mT5-large
# model.gradient_checkpointing_enable()
# logger.info("Model loaded in bf16 with gradient checkpointing enabled")

# LoRA Configuration
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=16,  
    lora_alpha=32,  # Match 2x rank
    lora_dropout=0.1,
    target_modules=["q", "v"],  # Attention query and value layers
    bias="none",
)

model = get_peft_model(model, peft_config)
if accelerator.is_main_process:
    model.print_trainable_parameters()

# --- 5. Training Arguments ---
training_args = Seq2SeqTrainingArguments(
    output_dir="./checkpoints/gec_german_mt5_large",
    eval_strategy="steps",
    eval_steps=2000,
    save_strategy="steps",
    save_steps=2000,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE * 2,  # Larger for eval (no gradients)
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    warmup_ratio=WARMUP_RATIO,
    weight_decay=0.01,
    max_grad_norm=1.0,  # Standard clipping
    num_train_epochs=NUM_EPOCHS,
    logging_steps=50,
    logging_dir="./logs",
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="bleu",
    greater_is_better=True,
    predict_with_generate=True,
    generation_max_length=MAX_LENGTH,
    generation_num_beams=1,  # Greedy for speed
    bf16=True,  # A6000 supports bf16
    fp16=False,
    optim="adamw_torch_fused",  # Fused optimizer is faster
    seed=SEED,
    dataloader_num_workers=2,  # Some parallelism helps with DDP
    dataloader_pin_memory=True,  # Faster CPU->GPU transfer
    push_to_hub=False,
    report_to=["none"],
    ddp_find_unused_parameters=False,  # Required False for gradient checkpointing
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
output_dir = os.path.join(os.path.dirname(__file__), "models/gec_german_mt5_large")
if trainer.is_world_process_zero():
    os.makedirs(output_dir, exist_ok=True)
    
trainer.save_model(output_dir)
logger.info(f"Model and Tokenizer saved to {output_dir}")