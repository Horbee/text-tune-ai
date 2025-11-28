# Interactive text correction script
# Loads the model and continuously corrects user input until exit

from transformers import T5Tokenizer, MT5ForConditionalGeneration
from peft import PeftModel
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'gec_german_mt5_lora')
BASE_MODEL = "google/mt5-base"

# Load model and tokenizer
print("Loading model...")
tokenizer = T5Tokenizer.from_pretrained(BASE_MODEL, use_fast=False, legacy=False)
base_model = MT5ForConditionalGeneration.from_pretrained(BASE_MODEL)
model = PeftModel.from_pretrained(base_model, MODEL_PATH)
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
