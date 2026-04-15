"""
Over-Correction Benchmark Module.

Measures the False Positive Rate (FPR) - how often a model "fixes"
a perfectly correct sentence when nothing is wrong.

This is crucial for GEC safety:
- Under-correction: Missing real errors (acceptable)
- Over-correction: Changing correct text (dangerous - can corrupt user content)
"""

import json
from dataclasses import dataclass
from typing import Optional
from src.inference import correct_text_latest


@dataclass
class FPRResult:
    sentence: str
    model_output: str
    was_modified: bool
    is_over_correction: bool
    confidence: float


def measure_fpr(
    model: str,
    flawless_sentences: list[dict],
    modification_threshold: float = 0.1,
    temperature: float = 0.0,
) -> dict:
    """
    Measure False Positive Rate on flawless sentences.

    Args:
        model: Ollama model name
        flawless_sentences: List of dicts with 'original' key containing correct sentences
        modification_threshold: Minimum edit distance ratio to consider as "modified"
        temperature: Model temperature (0 for deterministic)

    Returns:
        dict with FPR metrics and per-sentence results
    """
    results = []
    over_corrections = 0
    total = len(flawless_sentences)

    for item in flawless_sentences:
        sentence = item["original"]

        try:
            model_output = correct_text_latest(model, sentence).strip()

            is_modified = _detect_modification(
                sentence, model_output, modification_threshold
            )
            is_over_correction = is_modified

            if is_over_correction:
                over_corrections += 1

            results.append(
                FPRResult(
                    sentence=sentence,
                    model_output=model_output,
                    was_modified=is_modified,
                    is_over_correction=is_over_correction,
                    confidence=1.0,
                )
            )

        except Exception as e:
            results.append(
                FPRResult(
                    sentence=sentence,
                    model_output=f"ERROR: {str(e)}",
                    was_modified=False,
                    is_over_correction=False,
                    confidence=0.0,
                )
            )

    fpr = (over_corrections / total * 100) if total > 0 else 0.0

    return {
        "model": model,
        "total_sentences": total,
        "over_corrections": over_corrections,
        "fpr_percentage": round(fpr, 2),
        "safety_rating": _get_safety_rating(fpr),
        "per_sentence_results": [
            {
                "sentence": r.sentence,
                "model_output": r.model_output,
                "was_modified": r.was_modified,
                "is_over_correction": r.is_over_correction,
            }
            for r in results
        ],
    }


def _detect_modification(original: str, modified: str, threshold: float = 0.1) -> bool:
    """
    Detect if the model significantly modified the sentence.
    Uses normalized edit distance.
    """
    if original.strip() == modified.strip():
        return False

    import difflib

    matcher = difflib.SequenceMatcher(None, original, modified)
    similarity = matcher.ratio()

    return (1.0 - similarity) > threshold


def _get_safety_rating(fpr: float) -> str:
    """
    Convert FPR percentage to safety rating.
    """
    if fpr < 5:
        return "EXCELLENT - Very safe, minimal over-correction"
    elif fpr < 15:
        return "GOOD - Low over-correction tendency"
    elif fpr < 30:
        return "MODERATE - Some over-correction, use with caution"
    elif fpr < 50:
        return "POOR - High over-correction, requires human review"
    else:
        return "UNSAFE - Do not use without human oversight"


def run_fpr_benchmark(
    model: str, flawless_sentences: list[dict], output_file: Optional[str] = None
) -> dict:
    """
    Run the full FPR benchmark and optionally save results.
    """
    print(f"\n{'=' * 60}")
    print(f"OVER-CORRECTION BENCHMARK (False Positive Rate)")
    print(f"{'=' * 60}")
    print(f"Model: {model}")
    print(f"Test sentences: {len(flawless_sentences)}")
    print(f"{'-' * 60}")

    results = measure_fpr(model, flawless_sentences)

    print(f"\nRESULTS:")
    print(f"  False Positive Rate: {results['fpr_percentage']}%")
    print(
        f"  Over-corrections: {results['over_corrections']}/{results['total_sentences']}"
    )
    print(f"  Safety Rating: {results['safety_rating']}")

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to: {output_file}")

    return results
