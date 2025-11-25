#!/usr/bin/env python3
"""
Optimized mT5 training script for German Grammar Error Correction (GEC).
Includes memory optimization, mixed precision, gradient accumulation, and better monitoring.
"""

from datasets import Dataset, DatasetDict
from transformers import (
    T5Tokenizer,
    MT5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import evaluate
import numpy as np
from sklearn.model_selection import train_test_split
from peft import LoraConfig, get_peft_model, TaskType
import os
import pandas as pd
import torch
import logging

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
MODEL_CHECKPOINT = "google/mt5-base"
MAX_LENGTH = 128
BATCH_SIZE = 16  # Increased for efficiency
GRADIENT_ACCUMULATION_STEPS = 2  # Effective batch = 32
LEARNING_RATE = 1e-3  # Higher LR with LoRA
WARMUP_RATIO = 0.1
NUM_EPOCHS = 10
SEED = 42

# Check GPU availability
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {DEVICE}")
if DEVICE == "cuda":
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# --- 1. Load & Split Dataset ---
logger.info("Loading dataset...")
df = pd.read_json(
    # os.path.join(os.path.dirname(__file__), "data/chapter1-eval-v4.jsonl"),
    os.path.join(os.path.dirname(__file__), "data/news-data-train-v4.jsonl"),
    lines=True
)
logger.info(f"Total samples: {len(df)}")

train_df, val_df = train_test_split(
    df, test_size=0.15, random_state=SEED
)  # 85/15 split
logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}")

dataset = DatasetDict(
    {
        "train": Dataset.from_pandas(train_df).remove_columns(["__index_level_0__"]),
        "validation": Dataset.from_pandas(val_df).remove_columns(["__index_level_0__"]),
    }
)

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
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q", "v"],  # Attention query and value layers
    bias="none",
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# --- 5. Training Arguments ---
training_args = Seq2SeqTrainingArguments(
    output_dir="./checkpoints/gec_german_mt5_optimized",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE * 2,  # Can use larger for eval
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    warmup_ratio=WARMUP_RATIO,
    weight_decay=0.01,
    max_grad_norm=1.0,
    num_train_epochs=NUM_EPOCHS,
    logging_steps=50,
    logging_dir="./logs",
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="bleu",
    greater_is_better=True,
    predict_with_generate=True,
    generation_max_length=MAX_LENGTH,
    generation_num_beams=2,  # Lightweight beam search
    bf16=(DEVICE == "cuda" and torch.cuda.is_bf16_supported()),  # Use bf16 if available
    fp16=False,
    optim="adamw_torch",
    seed=SEED,
    dataloader_num_workers=0,  # Parallel data loading
    dataloader_pin_memory=False,
    push_to_hub=False,
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

# --- 8. Save Model ---
output_dir = os.path.join(os.path.dirname(__file__), "models/gec_german_mt5_optimized")
os.makedirs(output_dir, exist_ok=True)
trainer.save_model(output_dir)
logger.info(f"Model saved to {output_dir}")

# Save tokenizer
tokenizer.save_pretrained(output_dir)
logger.info("Tokenizer saved.")