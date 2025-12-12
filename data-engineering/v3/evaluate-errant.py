import errant
import spacy
import pandas as pd
import torch
from tqdm import tqdm
from transformers import T5Tokenizer, MT5ForConditionalGeneration
import argparse

parser = argparse.ArgumentParser(
            prog='evaluate-errant',
            description='Evaluate a text correction model with ERRANT for German GEC',
            usage='python evaluate-errant.py ./models/gec_german_mt5',
        )

parser.add_argument('model_name')
args = parser.parse_args()

class GermanGECEvaluator:
    def __init__(self):
        print("Loading German Spacy model...")
        # Load the large German model for better alignment
        self.nlp = spacy.load("de_core_news_lg")
        
        # Initialize ERRANT annotator
        # We force the language to 'de' (generic) so it uses the passed nlp object
        self.annotator = errant.load('en', nlp=self.nlp) 

    def extract_edits(self, original: str, corrected: str):
        """
        Aligns original and corrected text to find the edits.
        
        FIX: We perform manual Alignment and Merging but SKIP Classification.
        The default .annotate() method tries to classify errors (e.g., 'R:VERB'),
        which crashes with German POS tags like 'ART' (KeyError).
        """
        orig = self.annotator.parse(original)
        cor = self.annotator.parse(corrected)
        
        # Step 1: Align (Find matching tokens)
        alignment = self.annotator.align(orig, cor)
        
        # Step 2: Merge (Combine tokens into edits)
        # We use the merger directly to get edit objects.
        # These edits will have type "UNK" (Unknown), which is fine 
        # because we only ignore the type for F0.5 calculation.
        edits = self.annotator.merge(alignment)
        
        # Note: We purposefully DO NOT call self.annotator.classify(edits)
        return edits

    def evaluate_batch(self, sources, hypotheses, references):
        """
        sources: List of original (incorrect) sentences
        hypotheses: List of sentences your model generated
        references: List of gold standard (human corrected) sentences
        """
        
        tp = 0 # True Positives (Model made the right edit)
        fp = 0 # False Positives (Model made an edit, but it was wrong)
        fn = 0 # False Negatives (Model missed an error)

        print(f"Evaluating {len(sources)} sentences...")

        for src_text, hyp_text, ref_text in zip(sources, hypotheses, references):
            
            # 1. Get Gold Edits (Source vs Reference)
            gold_edits = self.extract_edits(src_text, ref_text)
            
            # 2. Get Model Edits (Source vs Hypothesis)
            model_edits = self.extract_edits(src_text, hyp_text)

            # 3. Compare Edits
            # We simplify edits to (start_token, end_token, corrected_text) tuples 
            # to compare them regardless of the "Error Type" label.
            gold_set = set((e.o_start, e.o_end, e.c_str) for e in gold_edits)
            model_set = set((e.o_start, e.o_end, e.c_str) for e in model_edits)

            # Calculate stats for this sentence
            current_tp = len(model_set.intersection(gold_set))
            current_fp = len(model_set - gold_set)
            current_fn = len(gold_set - model_set)

            tp += current_tp
            fp += current_fp
            fn += current_fn

        return self.calculate_f0_5(tp, fp, fn)

    def calculate_f0_5(self, tp, fp, fn):
        """
        Calculates standard GEC metrics.
        F0.5 weighs Precision twice as much as Recall.
        """
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        beta = 0.5
        f0_5 = ((1 + beta**2) * p * r) / ((beta**2 * p) + r) if (p + r) > 0 else 0.0

        return {
            "TP": tp, "FP": fp, "FN": fn,
            "Precision": round(p * 100, 2),
            "Recall": round(r * 100, 2),
            "F0.5": round(f0_5 * 100, 2)
        }

# --- Example Usage ---
if __name__ == "__main__":
    df = pd.read_json('data/chapter1-eval-v4.jsonl', lines=True)
    sources = df['de_corrupted'].tolist()
    references = df['de_correct'].tolist()

    tokenizer = T5Tokenizer.from_pretrained(args.model_name, legacy=False, use_fast=True)
    model = MT5ForConditionalGeneration.from_pretrained(args.model_name)

    tokenized_inputs = tokenizer(
        sources,
        max_length=128,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )

    model.eval()
    if torch.cuda.is_available():
        model.to('cuda')
        tokenized_inputs = {k: v.to('cuda') for k, v in tokenized_inputs.items()}

    # call in batches of 8
    BATCH_SIZE = 16
    all_outputs = []
    for i in tqdm(range(0, len(sources), BATCH_SIZE)):
        batch_input_ids = tokenized_inputs['input_ids'][i:i+BATCH_SIZE]
        batch_attention_mask = tokenized_inputs['attention_mask'][i:i+BATCH_SIZE]
        
        batch_output = model.generate(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask,
            max_length=128,
            num_beams=5, 
            early_stopping=True,
            repetition_penalty=2.5
        )
        
        all_outputs.extend(batch_output)
   
    # Make predictions with the model (hypotheses)
    hypotheses = [tokenizer.decode(ids, skip_special_tokens=True) for ids in all_outputs]

    # 2. Run Evaluation
    evaluator = GermanGECEvaluator()
    results = evaluator.evaluate_batch(sources, hypotheses, references)

    print("\n--- Final Results ---")
    print(results)