import torch
from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend
from peft import PeftModel

# --- Configuration ---
BASE_MODEL_NAME = "unsloth/Ministral-3-3B-Instruct-2512"

# Path to your saved checkpoint (the folder containing adapter_model.bin)
# e.g., "./results/checkpoint-500" or "./final_gec_model"
LORA_CHECKPOINT_PATH = "./models/Ministral-3-3B-GEC-v2" 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_lora_model():
    print(f"1. Loading Base Model: {BASE_MODEL_NAME}...")
    # Load the original heavy model
    # Note: We load it in 8-bit or half precision if possible to save memory,
    # but strictly speaking, standard float32 is safest for inference compatibility.
    base_model = Mistral3ForConditionalGeneration.from_pretrained(BASE_MODEL_NAME)
    
    print(f"2. Loading LoRA Adapters from: {LORA_CHECKPOINT_PATH}...")
    # This acts as a wrapper. It takes the base_model and overlays the adapters.
    model = PeftModel.from_pretrained(base_model, LORA_CHECKPOINT_PATH)
    
    # Merge adapters into the base model (Optional but faster for inference)
    model = model.merge_and_unload() 
    
    model.to(device)
    model.eval()
    
    print("3. Loading Tokenizer...")
    # Tokenizer is usually the same as the base model
    tokenizer = MistralCommonBackend.from_pretrained(BASE_MODEL_NAME)
    
    return model, tokenizer


def format_message(user_input: str):
    SYSTEM_PROMPT = "Korrigiere die Grammatik im folgenden Satz auf Standarddeutsch. Gib **nur** den korrigierten Satz zurück, ohne Anmerkungen."

    formatted_text = (
                    f"<s>[INST] {SYSTEM_PROMPT}\n\n"
                    f"{user_input} [/INST]"
                )
    
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
    result = result.strip().replace("[/INST]", "")
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

    model.save_pretrained("./models/Ministral-3-3B-GEC-v2-merged")
    tokenizer.save_pretrained("./models/Ministral-3-3B-GEC-v2-merged")