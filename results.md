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


Text-Tune-Small-v8 -> v9 -> v9 on eval v9 -> v9 ov eval v10 (3b) -> v10 ov eval v10 (3b)
========================================
1. Grammar Score: 54.5% -> 61.2% -> 73.8% -> 76.6% -> 80.2% -> 78.8%
2. Meaning Score: 58.4% -> 71.3% -> 74.4% -> 79.9% -> 87.3% -> 86.2%
3. STRICT ACCURACY: 46.8% -> 51.2% -> 64.2% -> 68.3% -> 77.1% -> 76.3%
4. GLEU: 0.7698 -> 0.8344 -> 0.8544 -> 0.8925 -> 0.9214 -> 0.9280
5. WER: 13.05% -> 7.64% -> 6.74% -> 5.26% -> 3.68% -> 3.34%


Text-Tune-Base-v8 (8b)
========================================
1. Grammar Score: 79.1% -> 79.6% -> 74.9% -> 87.6% -> 85.3%
2. Meaning Score: 83.7% -> 84.0% -> 81.5% -> 89.3% -> 88.1%
3. STRICT ACCURACY: 69.4% -> 66.1% -> 82.2% -> 81.4%
4. GLEU: 0.8826 -> 0.8955 -> 0.8796 -> 0.9446 -> 0.9375
5. WER: 5.53% -> 4.83% -> 5.53% -> 2.57% -> 2.84%


Text-Tune-Large-v8 (14b)
========================================
1. Grammar Score: 80.2% -> 75.2% -> 88.1% -> 87.3%
2. Meaning Score: 83.5% -> 79.3% -> 91% -> 88.1%
3. STRICT ACCURACY: 69.4% -> 65.3% -> 83.6% -> 82.5%
4. GLEU: 0.8936 -> 0.8804 -> 0.9444 -> 0.9423
5. WER: 4.93% -> 5.63% -> 2.57% -> 2.69%
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

