import nltk
import textdistance
from nltk.translate.gleu_score import corpus_gleu
import pandas as pd

import argparse

parser = argparse.ArgumentParser(
            prog='scoring',
            description='Try out a text correction model',
            usage='python scoring.py ministral-3:3b-corrected.jsonl',
        )

parser.add_argument('input_file', type=str, help='Path to the input JSONL file, e.g., ministral-3:3b-corrected.jsonl')
args = parser.parse_args()

# Ensure you have the tokenizer
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def evaluate_batch(reference_list, hypothesis_list):
    """
    Calculates scientifically accurate WER and GLEU for a batch of sentences.
    
    Args:
        reference_list (list of str): The "Gold Standard" correct sentences.
        hypothesis_list (list of str): The model's corrected outputs.
    """
    
    total_distance = 0
    total_ref_length = 0
    
    # Pre-process for GLEU (Requires list of list of tokens)
    refs_tokenized = [] # Format: [[['word', 'word']], [['word', 'word']]]
    hyps_tokenized = [] # Format: [['word', 'word'], ['word', 'word']]
    
    for ref, hyp in zip(reference_list, hypothesis_list):
        # 1. Tokenize (German language support is important)
        r_toks = nltk.word_tokenize(ref, language="german")
        h_toks = nltk.word_tokenize(hyp, language="german")
        
        # 2. Accumulate stats for WER (Micro-Average)
        # Calculate Levenshtein distance on *words*, not characters
        dist = textdistance.levenshtein.distance(r_toks, h_toks)
        total_distance += dist
        total_ref_length += len(r_toks)
        
        # 3. Prepare data for GLEU
        # corpus_gleu expects a list of references for each hypothesis
        refs_tokenized.append([r_toks]) 
        hyps_tokenized.append(h_toks)
        
    # --- CALCULATE FINAL SCORES ---
    
    # 1. Calculate Micro-Average WER
    if total_ref_length > 0:
        corpus_wer = total_distance / total_ref_length
    else:
        corpus_wer = 0.0
        
    # 2. Calculate Corpus GLEU
    # This automatically handles the "summing" logic internally
    corpus_gleu_score = corpus_gleu(refs_tokenized, hyps_tokenized)
    
    return {
        "GLEU": corpus_gleu_score, 
        "WER": corpus_wer
    }

# --- EXAMPLE USAGE ---
# references = [
#     "Ich bin in den Park gegangen.",
#     "Das ist ein schönes Auto."
# ]
# model_outputs = [
#     "Ich bin in dem Park gegangen.", # Error: dem vs den
#     "Das ist ein schönes Auto."      # Perfect match
# ]

df = pd.read_json(args.input_file, lines=True)

# remove rows where 'corrected_text' is ""
df = df[df['corrected_text'] != ""]

references = df['original_text'].tolist()
model_outputs = df['corrected_text'].tolist()


scores = evaluate_batch(references, model_outputs)

print(f"Final System Score:")
print(f"GLEU: {scores['GLEU']:.4f} (Higher is better)")
print(f"WER:  {scores['WER']:.2%} (Lower is better)")