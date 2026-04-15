"""
Tiered Evaluator for Stratified GEC Evaluation.

Evaluates model performance on difficulty-tiered datasets using an LLM judge.
"""

import json
from ollama import Client
from pydantic import BaseModel


class TierEvalResult(BaseModel):
    original_sentence: str
    model_output: str
    is_grammatically_correct: bool
    meaning_preserved: bool
    reason: str


SYSTEM_PROMPT = """
You are an expert German linguist evaluating a Grammatical Error Correction (GEC) system.

Evaluate the MODEL OUTPUT on two criteria:
1. is_grammatically_correct: Is the MODEL OUTPUT completely free of grammatical, spelling, case (Kasus), and punctuation errors? (true/false)
2. meaning_preserved: Does the MODEL OUTPUT preserve the intended meaning of the CORRUPTED sentence while fixing its grammar to match the standard of the ORIGINAL sentence? It should not delete crucial information or hallucinate new facts. (true/false)

OUTPUT FORMAT:
Provide the output strictly as a JSON array of objects. Do not include any conversational filler. Each object must have the following structure:
[
  {
    "original_sentence": "The exact input string of the original sentence",
    "model_output": "The exact input string of the model's corrected output",
    "is_grammatically_correct": true/false,
    "meaning_preserved": true/false,
    "reason": "A brief 3-5 word explanation for the decision"
  }
]
"""


def _correct_texts_batch(
    model: str,
    corrupted_texts: list[str],
) -> list[str]:
    """Get model corrections for a batch of corrupted texts."""
    from src.inference import correct_text_latest

    corrections = []
    for text in corrupted_texts:
        try:
            # Using correct_text_latest by default for structured output
            # If original models are needed, this could be changed to correct_text_original
            corrected = correct_text_latest(model, text)
            corrections.append(corrected)
        except Exception as e:
            corrections.append(f"ERROR: {str(e)}")
    return corrections


def _evaluate_batch(
    original_sentences: list[str],
    corrupted_sentences: list[str],
    model_outputs: list[str],
    client: Client,
    judge_model: str,
    temperature: float = 0.0,
) -> list[dict]:
    """Evaluate a batch using the LLM judge."""
    formatted_input = "\n".join(
        [
            f'CORRUPTED: "{corr}" ORIGINAL: "{orig}" MODEL OUTPUT: "{output}"'
            for corr, orig, output in zip(
                corrupted_sentences, original_sentences, model_outputs
            )
        ]
    )

    try:
        response = client.chat(
            model=judge_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": formatted_input},
            ],
            options={"temperature": temperature},
            format=TierEvalResult.model_json_schema(),
        )

        result_str = response["message"]["content"]
        return json.loads(result_str)

    except Exception as e:
        print(f"Error evaluating batch: {e}")
        return [
            {
                "original_sentence": orig,
                "model_output": output,
                "is_grammatically_correct": False,
                "meaning_preserved": False,
                "reason": f"ERROR: {str(e)[:50]}",
            }
            for orig, output in zip(original_sentences, model_outputs)
        ]


def evaluate_tier(
    model: str,
    tier_data: list[dict],
    ollama_client: Client,
    judge_model: str = "gpt-oss:120b-cloud",
    temperature: float = 0.0,
) -> dict:
    """
    Evaluate a model's performance on a specific tier.

    Steps:
    1. Get model's corrections for corrupted texts
    2. Judge corrections against original texts
    """
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.utils import process_in_batches

    original_sentences = [item["original"] for item in tier_data]
    corrupted_sentences = [item["corrupted"] for item in tier_data]

    all_corrections = []
    for batch in process_in_batches(corrupted_sentences, batch_size=10, verbose=False):
        corrections = _correct_texts_batch(model, batch)
        all_corrections.extend(corrections)

    results = []
    correct_count = 0
    grammar_count = 0
    meaning_count = 0

    for batch_orig, batch_corr, batch_model_out in zip(
        process_in_batches(original_sentences, batch_size=10, verbose=False),
        process_in_batches(corrupted_sentences, batch_size=10, verbose=False),
        process_in_batches(all_corrections, batch_size=10, verbose=False),
    ):
        batch_results = _evaluate_batch(
            batch_orig,
            batch_corr,
            batch_model_out,
            ollama_client,
            judge_model,
            temperature,
        )
        results.extend(batch_results)

        for r in batch_results:
            if r["is_grammatically_correct"]:
                grammar_count += 1
            if r["meaning_preserved"]:
                meaning_count += 1
            if r["is_grammatically_correct"] and r["meaning_preserved"]:
                correct_count += 1

    total = len(tier_data)
    tier_num = tier_data[0].get("difficulty", "unknown") if tier_data else "unknown"

    return {
        "tier": tier_num,
        "total_sentences": total,
        "correct_count": correct_count,
        "grammar_accuracy": round(grammar_count / total * 100, 1) if total > 0 else 0,
        "meaning_accuracy": round(meaning_count / total * 100, 1) if total > 0 else 0,
        "overall_accuracy": round(correct_count / total * 100, 1) if total > 0 else 0,
        "per_sentence_results": [
            {**r, "corrupted_sentence": corr}
            for r, corr in zip(results, corrupted_sentences)
        ],
    }


def evaluate_all_tiers(
    model: str,
    tier_data: list[dict],
    ollama_client: Client,
    judge_model: str = "gpt-oss:120b-cloud",
) -> dict:
    """
    Evaluate a model across all difficulty tiers.
    """
    tier1_data = [d for d in tier_data if d.get("difficulty") == "easy"]
    tier2_data = [d for d in tier_data if d.get("difficulty") == "medium"]
    tier3_data = [d for d in tier_data if d.get("difficulty") == "hard"]

    tier_results = {
        "tier_1": evaluate_tier(model, tier1_data, ollama_client, judge_model),
        "tier_2": evaluate_tier(model, tier2_data, ollama_client, judge_model),
        "tier_3": evaluate_tier(model, tier3_data, ollama_client, judge_model),
    }

    total = sum(r["total_sentences"] for r in tier_results.values())
    total_correct = sum(r["correct_count"] for r in tier_results.values())

    return {
        "model": model,
        "tier_1": tier_results["tier_1"],
        "tier_2": tier_results["tier_2"],
        "tier_3": tier_results["tier_3"],
        "summary": {
            "overall_accuracy": round(total_correct / total * 100, 1)
            if total > 0
            else 0,
            "tier_1_accuracy": tier_results["tier_1"]["overall_accuracy"],
            "tier_2_accuracy": tier_results["tier_2"]["overall_accuracy"],
            "tier_3_accuracy": tier_results["tier_3"]["overall_accuracy"],
        },
    }


def save_tier_results(results: dict, output_file: str) -> None:
    """Save tier evaluation results to a JSON file."""
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
