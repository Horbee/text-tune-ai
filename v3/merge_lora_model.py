import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# --- Configuration ---
BASE_MODEL_NAME = "google/mt5-base"

# Path to your saved checkpoint (the folder containing adapter_model.bin)
# e.g., "./results/checkpoint-500" or "./final_gec_model"
LORA_CHECKPOINT_PATH = "./models/gec_german_mt5_lora" 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_lora_model():
    print(f"1. Loading Base Model: {BASE_MODEL_NAME}...")
    # Load the original heavy model
    # Note: We load it in 8-bit or half precision if possible to save memory,
    # but strictly speaking, standard float32 is safest for inference compatibility.
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_MODEL_NAME,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    print(f"2. Loading LoRA Adapters from: {LORA_CHECKPOINT_PATH}...")
    # This acts as a wrapper. It takes the base_model and overlays the adapters.
    model = PeftModel.from_pretrained(base_model, LORA_CHECKPOINT_PATH)
    
    # Merge adapters into the base model (Optional but faster for inference)
    model = model.merge_and_unload() 
    
    model.to(device)
    model.eval()
    
    print("3. Loading Tokenizer...")
    # Tokenizer is usually the same as the base model
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, legacy=False)
    
    return model, tokenizer

def correct_grammar(model, tokenizer, text):
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        max_length=128, 
        truncation=True
    ).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_length=128, 
            num_beams=5,        # Beam search for better quality
            early_stopping=True,
            length_penalty=1.0
        )
        
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
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

    model.save_pretrained("./models/gec_german_mt5")
    tokenizer.save_pretrained("./models/gec_german_mt5")