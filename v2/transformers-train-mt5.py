# pip install transformers evaluate numpy scikit-learn pandas peft sentencepiece hf_transfer protobuf sacrebleu

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    DataCollatorForSeq2Seq, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer
)
import evaluate
import numpy as np
from sklearn.model_selection import train_test_split
import os
import pandas as pd
from peft import LoraConfig, get_peft_model, TaskType

# --- Configuration ---
MODEL_CHECKPOINT = "google/mt5-base" # Multilingual T5
MAX_LENGTH = 128
PREFIX = "korrigiere: " # Task prefix helps the model understand the goal
BATCH_SIZE = 16  # Adjust based on your GPU VRAM

# --- 1. Load Dataset ---
# Assuming you have local JSON files. 
# If you don't have files yet, create dummy ones to test this script.
# Load single data file and split into train/validation
df = pd.read_json(os.path.join(os.path.dirname(__file__), 'data/news_data_text_5000_corrupted.jsonl'), lines=True)
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Create dataset dictionary
dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df),
    "validation": Dataset.from_pandas(val_df)
})

# --- 2. Tokenizer & Preprocessing ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, use_fast=False, legacy=False)

def preprocess_function(examples):
    # Add prefix to the inputs
    inputs = [PREFIX + doc for doc in examples["de_corrupted"]]
    targets = [doc for doc in examples["de_correct"]]

    # Tokenize inputs
    model_inputs = tokenizer(
        inputs, 
        max_length=MAX_LENGTH, 
        truncation=True
    )

    # Tokenize targets (labels)
    labels = tokenizer(
        text_target=targets, 
        max_length=MAX_LENGTH, 
        truncation=True
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply processing to dataset
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# --- 3. Metrics (BLEU) ---
# We use BLEU for monitoring during training because it's fast.
metric = evaluate.load("sacrebleu")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    
    # Clip predictions to valid token range
    vocab_size = len(tokenizer)
    preds = np.where(preds < vocab_size, preds, tokenizer.pad_token_id)
    preds = np.where(preds >= 0, preds, tokenizer.pad_token_id)
    
    # Decode generated predictions
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Post-processing: trim spaces
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": result["score"]}

# --- 4. Model & Training ---
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)

# Define LoRA Config
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM, 
    inference_mode=False, 
    r=8,            # Rank (larger = more parameters to train, but better performance)
    lora_alpha=32,  # Scaling factor
    lora_dropout=0.1
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

args = Seq2SeqTrainingArguments(
    output_dir="./gec_german_mt5_model",
    eval_strategy="epoch",
    # eval_strategy="steps",
    # eval_steps=4000, 
    learning_rate=5e-4, # 1e-3  # Increased learning rate
    per_device_train_batch_size=BATCH_SIZE, # Adjust based on VRAM
    per_device_eval_batch_size=BATCH_SIZE,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=5,
    predict_with_generate=True, # Essential for calculating BLEU during training
    fp16=False, # Disabled - causes NaN with mT5
    bf16=True,  # Enable bfloat16 for better performance on TPUs and some GPUs
    logging_steps=300,  # Log every 10 steps
    logging_dir="./logs",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="bleu",
    greater_is_better=True,
    push_to_hub=False,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Start Training
print("Starting training...")
trainer.train()

# Save final model
trainer.save_model(f"{os.path.join(os.path.dirname(__file__), 'final_gec_mt5_model2')}")
print("Model saved.")