import torch
from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend
from peft import PeftModel

import argparse

parser = argparse.ArgumentParser(
            prog='merge_lora_model',
            description='This script merges a LoRA fine-tuned model with its base Mistral model for inference.',
            usage='python merge_lora_model.py --base_model_path BASE_MODEL_PATH --lora_checkpoint_path LORA_CHECKPOINT_PATH --output_path OUTPUT_PATH',
        )

parser.add_argument('base_model_path', type=str, help='Path to the base Mistral model, e.g., unsloth/Ministral-3-3B-Instruct-2512')
parser.add_argument('lora_checkpoint_path', type=str, help='Path to the LoRA checkpoint, e.g., ./models/Ministral-3-3B-GEC-v2')
parser.add_argument('output_path', type=str, help='Path to save the merged model')
args = parser.parse_args()


# --- Configuration ---
BASE_MODEL_NAME = args.base_model_path

# Path to your saved checkpoint (the folder containing adapter_model.bin)
LORA_CHECKPOINT_PATH = args.lora_checkpoint_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_lora_model():
    print(f"1. Loading Base Model: {BASE_MODEL_NAME}...")
    base_model = Mistral3ForConditionalGeneration.from_pretrained(BASE_MODEL_NAME, 
                                                                #  torch_dtype=torch.float16,
                                                                 device_map="auto",
                                                                 offload_buffers=True
                                                                #  low_cpu_mem_usage=True
                                                                 )
    
    print(f"2. Loading LoRA Adapters from: {LORA_CHECKPOINT_PATH}...")
    # This acts as a wrapper. It takes the base_model and overlays the adapters.
    model = PeftModel.from_pretrained(base_model, LORA_CHECKPOINT_PATH, offload_buffers=True)
    
    # Merge adapters into the base model (Optional but faster for inference)
    model = model.merge_and_unload() 
    
    model.to(device)
    model.eval()
    
    print("3. Loading Tokenizer...")
    # Tokenizer is usually the same as the base model
    tokenizer = MistralCommonBackend.from_pretrained(BASE_MODEL_NAME)
    
    return model, tokenizer


def format_message(user_input: str):
    # SYSTEM_PROMPT = "Korrigiere die Grammatik im folgenden Satz auf Standarddeutsch. Gib **nur** den korrigierten Satz zurück, ohne Anmerkungen."
    SYSTEM_PROMPT = "Korrigiere die Grammatik im folgenden Text, aber behalte den ursprünglichen Stil und Ton bei. Verleihe dem Text keine formelle Note, wenn er diese nicht hat. Gib **nur** den korrigierten Satz zurück, ohne Anmerkungen. Wenn der Satz korrekt ist, gib ihn unverändert zurück."

    formatted_text = f"<s>[SYSTEM_PROMPT]{SYSTEM_PROMPT}[/SYSTEM_PROMPT][INST]{user_input}[/INST]"
    
    return formatted_text


def correct_grammar(model, tokenizer, text):
    inputs = tokenizer(
        format_message(text), 
        return_tensors="pt", 
        max_length=4096, 
    ).to(device)
    
    with torch.inference_mode():
        outputs = model.generate(
            **inputs, 
            max_length=4096, 
        )

    result = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    # result = result.strip().replace("[/INST]", "")
    return result

if __name__ == "__main__":
    # Load everything
    model, tokenizer = load_lora_model()
    
    # Test Sentences
    test_sentences = [
        "ich gehe schule weil wetter gut ist",  # Broken
        "Das ist ein fehler.",                 # Capitalization
        "Wir haben kein geld für den bus."      # Noun capitalization
    ]
    
    print("\n--- Inference Tests ---")
    for sentence in test_sentences:
        corrected = correct_grammar(model, tokenizer, sentence)
        print(f"Original:  {sentence}")
        print(f"Corrected: {corrected}")
        print("-" * 30)


    model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)