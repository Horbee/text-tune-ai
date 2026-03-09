from transformers import MistralCommonBackend, Mistral3ForConditionalGeneration 
from peft import PeftModel
import argparse
import torch

parser = argparse.ArgumentParser(
            prog='inference',
            description='Try out a text correction model',
            usage='python inference.py unsloth/Ministral-3-8B-Instruct-2512 --peft ./models/Ministral-3-8B-GEC-v1',
            # usage='python inference.py unsloth/Ministral-3-3B-Instruct-2512 --peft ./models/Ministral-3-3B-GEC-v2',
        )

parser.add_argument('model_path', type=str, help='Huggingface Url or path to the fine-tuned model')
parser.add_argument('--peft', type=str, default=None, help='Path to the saved adapters if using PEFT')
args = parser.parse_args()

def format_messages(user_input: str):
    SYSTEM_PROMPT = "Korrigiere die Grammatik im folgenden Satz auf Standarddeutsch. Gib **nur** den korrigierten Satz zurück, ohne Anmerkungen."
    # user_input = user_input.strip().lower() # should I lowercase? KAtze -> Katze, NO augment your training data with synthetic noise.

    formatted_text = (
                    f"<s>[INST] {SYSTEM_PROMPT}\n\n"
                    f"{user_input} [/INST] " # trailing space is important!!!
                )
    
    return formatted_text

# Load model and tokenizer
def load_model(peft: str = None):
    print("Loading model...")
    if (peft):
        tokenizer = MistralCommonBackend.from_pretrained(args.model_path)
        base_model = Mistral3ForConditionalGeneration.from_pretrained(args.model_path)
        model = PeftModel.from_pretrained(base_model, peft)
    else:
        tokenizer = MistralCommonBackend.from_pretrained(args.model_path)
        model = Mistral3ForConditionalGeneration.from_pretrained(args.model_path)
    
    return model, tokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
model, tokenizer = load_model(args.peft)
model.to(device)
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
    
    inputs = tokenizer(format_messages(user_input), return_tensors="pt").to(device)
    
    outputs = model.generate(
        **inputs,
        max_length=4096,
    )
    
    result = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    result = result.strip().replace("[/INST]", "")
    print(result)