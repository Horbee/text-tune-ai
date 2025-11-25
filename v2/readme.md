# Iteration 2

- Datasets converted to JSONL
- Corruption strategy is more simple: max 2, divide them through the dataset
- training starts simple, model learns only a few fix rules
- Using the more robust mt5 model
- Evaluation is done with BLEU during training and a modified ERRANT for german at the end
- Train longer (5 epochs) with a higher LR
- used LoRA to train the model more efficiently
- used Runpod to utilize a cloud A6000 GPU for training
