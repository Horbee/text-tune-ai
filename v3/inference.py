# Interactive text correction script
# Loads the model and continuously corrects user input until exit

from transformers import T5Tokenizer, MT5ForConditionalGeneration
from peft import PeftModel
import os
import argparse

parser = argparse.ArgumentParser(
            prog='inference',
            description='Try out a text correction model',
            usage='python inference.py ./models/gec_german_mt5',
        )

parser.add_argument('model_name')
parser.add_argument('--peft', type=bool, default=False, help='Use PEFT model loading if True')
parser.add_argument('--base_model', type=str, default='google/mt5-base', help='Base model name or path')
args = parser.parse_args()

# Load model and tokenizer
def load_model(use_peft: bool):
    print("Loading model...")
    tokenizer = T5Tokenizer.from_pretrained(args.model_name, use_fast=False, legacy=False)
    if (use_peft):
        base_model = MT5ForConditionalGeneration.from_pretrained(args.base_model)
        model = PeftModel.from_pretrained(base_model, args.model_name)
    else:
        model = MT5ForConditionalGeneration.from_pretrained(args.model_name)
    
    return model, tokenizer

model, tokenizer = load_model(args.peft)
model.eval()
print("Model loaded successfully!")

print("\n" + "="*80)
print("German Grammar Error Correction - Interactive Mode")
print("="*80)
print("Enter text to correct, or type 'exit' or 'quit' to stop.\n")

while True:
    # Get user input
    user_input = input("Enter text: ").strip()
    
    # Check for exit commands
    if user_input.lower() in ['exit', 'quit', '']:
        print("Exiting...")
        break
    
    # Process the input
    inputs = tokenizer(user_input, return_tensors="pt", max_length=128, truncation=True)
    
    outputs = model.generate(
        **inputs,
        max_length=128,
        num_beams=4,
        early_stopping=True
    )
    
    corrected = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"Corrected:  {corrected}\n")
