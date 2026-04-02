# Suggestions for v11

- Increase Volume: Because the new filtering pipeline is so strict, we need to feed it more raw text. Try to scale the v10 dataset up to at least 7,000–8,000 samples to compensate for the lost volume.
- Inject Controlled Noise: In the corruption prompt, tell gpt-oss-120b that for 20% of the sentences, it should apply two or three errors instead of just one. This will push the average edit distance back up to around ~5.5, re-introducing the "difficulty" the model needs to build robust reasoning, without reverting to the meaning-destroying gibberish of v9.

# Iteration 9-10

- v9 models performed better (1-2%)
- getting performance boost from more data cleaning
- adjusting prompts for gpt-oss-120 to avoid broken sentences and grammar problems
- added high thinking to gpt-oss-120 
- results for 3b model were very good
- v9 Training Dataset:
- - Total Samples: 6,776
- - Average Sentence Length: ~99.7 characters
- - Average Edit Distance: 5.47 characters
- v10 Training Dataset:
- - Total Samples: 6,493
- - Average Sentence Length: ~97.8 characters
- - Average Edit Distance: 4.71 characters

The "Sterile Data" Problem (Loss of Regularization) - the corruptions were larger, noisier, and sometimes chaotic. The model had to work much harder to reconstruct the sentences, which acted as a powerful regularizer and made the model robust.

# Iteration 7 - 8

- scale up training data by collecting more from german wikipedia
- adjusting training scripts for better performance
- performance was suprisingly worse
- while inspecting data I've found quality problems in the training and eval data as well
- models were punished for good results: ex.: figuring out hallucinated corrupted sentences   

# Iteration 6

- train Ministral-3:3b, 8b and 14b
- use gpt-oss-120b to create synthetic corruption data
- dataset is clean 3.5k german comments
- use LLM Judge (gpt-oss-120b) to see if recovered sentences are grammatically correct and meaning is preserved
- use WER and GLEU as extra scores, but LLM strict accuracy from above is more meaningful

# Iteration 5

- Train an LLM with Unsloth
- Ministral-3:3b and 8b (good german knowledge)
- 10k news data
- synthetic corruptor script - sometimes context is lost, bad corruptions
- use WER and GLEU scores to evaluate
- deploy to Ollama with GGUF and Q4_K_M quantization
- fixed instruction formatting, becase model returned sometimes only the end of conversation token

# Iteration 3 and 4

- global control over how many errors of each type appear
- per-sentence intentional selection of 0–3 error types
- evenly balanced error categories
- maintain error diversity
- allow adding more clean sentences (none)
- realistic multi-error sentences

- trainer script is prepared for multi GPU training (accelerate)

## Workflow

- synthesize_errors_v4.py
- data_splitter.py
- transformers-train-mt5.py
- merge_lora_model.py
- bleu-test.ipynb - 'score': 87.96022085096091 
- evaluate-errant.py last results: {'TP': 214, 'FP': 66, 'FN': 180, 'Precision': 76.43, 'Recall': 54.31, 'F0.5': 70.67}

## Try out the model

- inference.py



# Iteration 2

- Datasets converted to JSONL
- Corruption strategy is more simple: max 2, divide them through the dataset
- training starts simple, model learns only a few fix rules
- Using the more robust mt5 model
- Evaluation is done with BLEU during training and a modified ERRANT for german at the end
- Train longer (5 epochs) with a higher LR
- used LoRA to train the model more efficiently
- used Runpod to utilize a cloud A6000 GPU for training


# Iteration 1

- Dataset in CSV format
- More aggressive text corruption strategy
- Using the t5-base model
- Evaluation only uses loss
