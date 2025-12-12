from transformers import T5Tokenizer, T5ForConditionalGeneration


MODELNAME = "t5-small"
PREFIX = "grammar: "


tokenizer = T5Tokenizer.from_pretrained(MODELNAME)
model = T5ForConditionalGeneration.from_pretrained(MODELNAME)

# Define the input text with a task prefix
input_text = PREFIX + "he go to school yesterday."

# Tokenize the input
input_ids = tokenizer.encode(
    input_text,
    return_tensors="pt",
    max_length=128,
    truncation=True
)

# Generate output (corrected text)
output = model.generate(
    input_ids,
    max_length=128,
    num_beams=5,  # Beam search for better quality
    early_stopping=True,
    repetition_penalty=2.5  # Penalize repetitive output
)

# Decode the generated text
corrected_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(f"Original: he go to school yesterday.")
print(f"Corrected: {corrected_text}")
