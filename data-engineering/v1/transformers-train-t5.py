# 1. Load Tokenizer and Model
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from datasets import Dataset, DatasetDict
import pandas as pd
from sklearn.model_selection import train_test_split
# from calc_length import plot_lengths
# import sys

MODEL_NAME = "google-t5/t5-base"
MAX_LENGTH = 160
BATCH_SIZE = 8

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# 2. Preprocessing Function
prefix = "grammar: "


def preprocess_function(examples):
    # Tokenize inputs (the faulty sentences with prefix)
    inputs = [prefix + doc for doc in examples["de_corrupted"]]
    model_inputs = tokenizer(inputs, max_length=MAX_LENGTH,
                             truncation=True, padding="max_length")

    # Tokenize targets (the correct sentences)
    # Set padding to -100 to make the model ignore padding tokens in the loss calculation
    labels = tokenizer(text_target=examples["de_correct"],
                       max_length=MAX_LENGTH, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# 3. Load and Process Dataset (assuming you have a HF Dataset object 'raw_datasets')
df = pd.read_csv("data/news_data_corrupted_50k.csv", sep=";")
df = df.dropna()
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_ds = Dataset.from_pandas(train_df)
test_ds = Dataset.from_pandas(test_df)

dataset = DatasetDict()

dataset['train'] = train_ds
dataset['test'] = test_ds

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Optional: Plot lengths to understand data distribution
# plot_lengths(tokenizer, tokenized_datasets["train"], prefix)
# sys.exit(0)

# 4. Define Training Arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./tf-results",
    eval_strategy="epoch",
    learning_rate=1e-4,  # T5 models often respond well to a slightly higher LR
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=3,
    predict_with_generate=True,  # Important for seq2seq tasks
    fp16=False,  
    bf16=True,  # More stable than fp16
)

# 5. Data Collator
# This will handle padding dynamically for inputs and labels
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# 6. Initialize Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# 7. Start Training
trainer.train()
