# Test script to verify model inference and BLEU calculation
# Run this after training to test your model

from transformers import T5Tokenizer, MT5ForConditionalGeneration
from peft import PeftModel
import evaluate
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'gec_german_mt5_lora')
BASE_MODEL = "google/mt5-base"

# Load model and tokenizer
print("Loading model...")
tokenizer = T5Tokenizer.from_pretrained(BASE_MODEL, use_fast=False, legacy=False)
base_model = MT5ForConditionalGeneration.from_pretrained(BASE_MODEL)
model = PeftModel.from_pretrained(base_model, MODEL_PATH)
model.eval()

# Load BLEU metric
metric = evaluate.load("sacrebleu")

# Test examples (corrupted -> correct)
test_cases = [
    {
        "input": "Ich gehe zu der Schule.",  # Article error
        "expected": "Ich gehe zu Schule."
    },
    {
        "input": "Er hat das gemacht das ich wollte.",  # dass vs das
        "expected": "Er hat das gemacht, dass ich wollte."
    },
    {
        "input": "Die schöne haus ist groß.",  # Adjective ending + article error
        "expected": "Das schöne Haus ist groß."
    },
]

print("\nTesting corrections:")
print("-" * 80)

all_preds = []
all_refs = []

for i, test in enumerate(test_cases, 1):
    input_text = test["input"]
    inputs = tokenizer(input_text, return_tensors="pt", max_length=128, truncation=True)
    
    outputs = model.generate(
        **inputs,
        max_length=128,
        num_beams=4,
        early_stopping=True
    )
    
    corrected = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"\nTest {i}:")
    print(f"  Input:    {test['input']}")
    print(f"  Output:   {corrected}")
    print(f"  Expected: {test['expected']}")
    
    all_preds.append(corrected)
    all_refs.append([test["expected"]])

# Calculate BLEU
result = metric.compute(predictions=all_preds, references=all_refs)
print(f"\n{'='*80}")
print(f"BLEU Score: {result['score']:.2f}")
print(f"{'='*80}")

# Additional diagnostic: Check if model is just copying input
unique_outputs = len(set(all_preds))
print(f"\nDiagnostics:")
print(f"  Unique outputs: {unique_outputs}/{len(all_preds)}")
print(f"  Model is {'NOT ' if unique_outputs > 1 else ''}copying inputs")
