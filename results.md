# Experiment v6

Synthetic data corrupted using GPT-OSS-120b model: `src/corruptor.py`

Train dataset size: 3.5k creative German (train / test split ratio is `0.9`)

## GEC Evaluations for v6

363 rows of creative German sentences. Corruption by GPT-OSS-120b model with the same prompt: `src/corruptor.py`

Eval metrics used:

1. Grammar Score: how many sentences are fixed by the model with perfect grammar
2. Meaning Score: how many sentences are fixed by the model without deletions/hallucinations
3. STRICT ACCURACY: where Grammar AND Meaning are perfect
4. GLEU: Single sentence evaluation where the minimum of precision and recall for 1, 2, 3, or 4-grams are measured. 1 represents a perfect match between the generated text and the reference, and 0 indicates no overlap.
5. WER: word error rate is derived from the Levenshtein distance (also known as edit distance), which counts the minimum number of operations needed to transform the system's output into the correct text. Lower is better. WER < 5-10%: Excellent / Commercial Grade

```py
Ministral-3:14b stock model # This model got the same prompt (v5) as the fine tuned models, just the returning ** symbols were removed.
========================================
1. Grammar Score : 81.5%
2. Meaning Score : 72.7%
3. STRICT ACCURACY : 61.7%
4. GLEU: 0.7289
5. WER: 15.69%

Text-Tune-Base-v4 # Legacy model - This model were trained on 10k news data, corrupted by corruption script v4. This script sometimes created incorrect corruptions, which sometimes led to hallucinations and was hard to recover context.
========================================
1. Grammar Score: 58.7%
2. Meaning Score: 69.1%
3. STRICT ACCURACY: 47.7%
4. GLEU: 0.7883
5. WER: 10.79%

# Actual trained models for this experiment
Text-Tune-Small-v6 (3b)
========================================
1. Grammar Score: 54.5%
2. Meaning Score: 58.4%
3. STRICT ACCURACY: 46.8%
4. GLEU: 0.7698
5. WER: 13.05%

Text-Tune-Base-v6 (8b)
========================================
1. Grammar Score: 79.1%
2. Meaning Score: 83.7%
3. STRICT ACCURACY: 69.4%
4. GLEU: 0.8826
5. WER: 5.53%

Text-Tune-Large-v6 (14b)
========================================
1. Grammar Score: 80.2%
2. Meaning Score: 83.5%
3. STRICT ACCURACY: 69.4%
4. GLEU: 0.8936
5. WER: 4.93%



Text-Tune-Base-v7 (8b)
========================================
1. Grammar Score: 79.1% -> 79.6%
2. Meaning Score: 83.7% -> 84.0%
3. STRICT ACCURACY: 69.4%
4. GLEU: 0.8826 -> 0.8955
5. WER: 5.53% -> 4.83%
```


Text-Tune-Base-v6 (8b) on new data
========================================
 🏆 GEC EVALUATION REPORT
========================================
Total Sentences Evaluated : 748
1. Grammar Score          : 65.4%  (Sentences with perfect grammar)
2. Meaning Score          : 74.5%  (Sentences without deletions/hallucinations)
----------------------------------------
🌟 STRICT ACCURACY        : 52.3%  (Grammar AND Meaning are perfect)
========================================

