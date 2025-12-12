# Iteration 3

✔ global control over how many errors of each type appear
✔ per-sentence intentional selection of 0–3 error types
✔ evenly balanced error categories
✔ maintain error diversity
✔ allow adding more clean sentences (none)
✔ realistic multi-error sentences

✔ trainer script is prepared for multi GPU training (accelerate)

## Workflow

- synthesize_errors_v4.py
- data_splitter.py
- transformers-train-mt5.py
- merge_lora_model.py
- bleu-test.ipynb
- evaluate-errant.py last results: {'TP': 214, 'FP': 66, 'FN': 180, 'Precision': 76.43, 'Recall': 54.31, 'F0.5': 70.67}

## Try out the model

- inference.py
