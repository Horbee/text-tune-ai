import sys
import torch
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Configuration
MODEL_NAME = sys.argv[1]
TASK_PREFIX = "grammar: "
BATCH_SIZE = 8
MAX_INPUT_LENGTH = 160
MAX_TARGET_LENGTH = 160

# Custom Dataset and Collator
class GermanGECDataset(Dataset):
    """
    A custom PyTorch Dataset for German Grammatical Error Correction.
    It tokenizes data on the fly.
    """
    def __init__(self, data: List[tuple], tokenizer: AutoTokenizer, prefix: str, max_input_len: int, max_target_len: int):
        self.data = data
        self.tokenizer = tokenizer
        self.prefix = prefix
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        incorrect_text, correct_text = self.data[index]
        
        # Prepare the input text (with prefix)
        input_text = self.prefix + incorrect_text

        # Tokenize the input
        # We don't pad here; the collator will handle it.
        tokenized_input = self.tokenizer(
            input_text,
            max_length=self.max_input_len,
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize the target
        # We don't pad here; the collator will handle it.
        tokenized_target = self.tokenizer(
            text_target=correct_text,
            max_length=self.max_target_len,
            truncation=True,
            return_tensors="pt"
        )

        # Squeeze to remove the batch dimension (which is 1)
        # The DataLoader will add it back.
        input_ids = tokenized_input["input_ids"].squeeze(0)
        attention_mask = tokenized_input["attention_mask"].squeeze(0)
        labels = tokenized_target["input_ids"].squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


class GECDataCollator:
    """
    A custom data collator. This is crucial for a manual loop.
    It pads batches dynamically to the longest sequence in that batch.
    """
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        
        # --- 1. Pad Input IDs and Attention Masks ---
        
        # Get all input_ids from the batch
        input_ids_list = [item["input_ids"] for item in batch]
        
        # Pad them to the longest sequence in this batch
        # The tokenizer's pad method is excellent for this.
        # It handles padding, truncation, and returns a tensor.
        padded_inputs = self.tokenizer.pad(
            {"input_ids": input_ids_list},
            padding="longest",
            return_tensors="pt"
        )
        
        # --- 2. Pad Labels ---
        
        # Get all labels from the batch
        labels_list = [item["labels"] for item in batch]
        
        # Pad them to the longest sequence in this batch
        padded_labels = self.tokenizer.pad(
            {"input_ids": labels_list},
            padding="longest",
            return_tensors="pt"
        ).get("input_ids") # We only want the input_ids from this
        
        # --- 3. Mask Padded Labels ---
        
        # This is CRITICAL. The model should NOT calculate loss
        # on padding tokens in the labels.
        # We replace the pad_token_id (e.g., 0) with -100.
        # PyTorch's CrossEntropyLoss automatically ignores -100.
        padded_labels[padded_labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": padded_inputs["input_ids"],
            "attention_mask": padded_inputs["attention_mask"],
            "labels": padded_labels
        }


# Evaluation Loop
def evaluate_epoch(model, dataloader, device):
    """One full evaluation pass."""
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    
    # We don't need to track gradients during evaluation
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating")
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            # Get the loss
            loss = outputs.loss
            total_loss += loss.item()

            # Update progress bar with current batch loss
            progress_bar.set_postfix({"batch_loss": f"{loss.item():.4f}"})
            
    return total_loss / len(dataloader)


# Main Execution
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


print(f"Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print(f"Loading model: {MODEL_NAME}")
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.to(device)


df = pd.read_csv("data/chapter1_corrupted.csv", sep=";")
data_tuples = list(zip(df['de_corrupted'].tolist(), df['de_correct'].tolist()))

gec_collator = GECDataCollator(tokenizer=tokenizer)

dataset = GermanGECDataset(
    data_tuples, tokenizer, TASK_PREFIX, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH
)

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    collate_fn=gec_collator,
    shuffle=False
)


val_loss = evaluate_epoch(model, dataloader, device)
print(f"Validation Loss: {val_loss:.4f}")