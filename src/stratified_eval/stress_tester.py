"""
Stress Testing Module for GEC Models.

Tests model behavior at the limits of their capability:
1. Input Length: Short to very long texts
2. High-Density Errors: Multiple errors packed into one sentence
3. Dialect/Slang: Non-standard German forms
4. Edge Cases: Unusual punctuation, mixed languages, etc.
"""

import json
from dataclasses import dataclass
from typing import Optional
from ollama import chat, ChatResponse


@dataclass
class StressTestResult:
    category: str
    input_text: str
    expected_output: Optional[str]
    model_output: str
    was_modified: bool
    is_correct: bool
    word_count: int
    error_count: int


def run_stress_tests(model: str, stress_data: dict, temperature: float = 0.0) -> dict:
    """
    Run comprehensive stress tests on a GEC model.

    Args:
        model: Ollama model name
        stress_data: Dictionary containing stress test categories
        temperature: Model temperature

    Returns:
        dict with stress test results per category
    """
    results = {
        "model": model,
        "length_tests": [],
        "high_density_tests": [],
        "dialect_tests": [],
        "summary": {},
    }

    print(f"\n{'=' * 60}")
    print(f"STRESS TESTING: {model}")
    print(f"{'=' * 60}")

    for length_test in stress_data.get("length", []):
        result = _run_length_test(model, length_test, temperature)
        results["length_tests"].append(result)
        print(
            f"  Length [{length_test['category']}]: {'[PASS]' if result['is_correct'] else '[FAIL]'}"
        )

    for density_test in stress_data.get("high_density", []):
        result = _run_high_density_test(model, density_test, temperature)
        results["high_density_tests"].append(result)
        print(
            f"  High-Density [{density_test['category']}]: {'[PASS]' if result['is_correct'] else '[FAIL]'}"
        )

    for dialect_test in stress_data.get("dialect", []):
        result = _run_dialect_test(model, dialect_test, temperature)
        results["dialect_tests"].append(result)
        print(
            f"  Dialect [{dialect_test['category']}]: {'[PASS]' if result['is_correct'] else '[FAIL]'}"
        )

    _compute_stress_summary(results)

    return results


def _run_length_test(model: str, test_item: dict, temperature: float) -> dict:
    """Test model behavior across different input lengths."""
    input_text = test_item["text"]
    word_count = test_item.get("word_count", len(input_text.split()))

    try:
        response: ChatResponse = chat(
            model=model,
            messages=[
                {"role": "user", "content": input_text},
            ],
        )
        model_output = response["message"]["content"].strip()

        is_modified = input_text.strip() != model_output.strip()
        is_correct = not is_modified

        return {
            "category": test_item["category"],
            "input_text": input_text,
            "expected_output": input_text,
            "model_output": model_output,
            "was_modified": is_modified,
            "is_correct": is_correct,
            "word_count": word_count,
            "error_count": 0,
            "status": "success",
        }
    except Exception as e:
        return {
            "category": test_item["category"],
            "input_text": input_text,
            "expected_output": input_text,
            "model_output": f"ERROR: {str(e)}",
            "was_modified": False,
            "is_correct": False,
            "word_count": word_count,
            "error_count": 0,
            "status": "error",
        }


def _run_high_density_test(model: str, test_item: dict, temperature: float) -> dict:
    """Test model behavior with high-density errors."""
    input_text = test_item["corrupted"]
    expected = test_item.get("original", input_text)
    category = test_item.get("category", "unknown")
    note = test_item.get("note", "")

    error_count = _count_errors_in_text(input_text, expected)

    try:
        response: ChatResponse = chat(
            model=model,
            messages=[
                {"role": "user", "content": input_text},
            ],
        )
        model_output = response["message"]["content"].strip()

        is_correct = _check_correction_quality(expected, model_output)

        return {
            "category": category,
            "note": note,
            "input_text": input_text,
            "expected_output": expected,
            "model_output": model_output,
            "was_modified": True,
            "is_correct": is_correct,
            "word_count": len(input_text.split()),
            "error_count": error_count,
            "status": "success",
        }
    except Exception as e:
        return {
            "category": category,
            "note": note,
            "input_text": input_text,
            "expected_output": expected,
            "model_output": f"ERROR: {str(e)}",
            "was_modified": False,
            "is_correct": False,
            "word_count": len(input_text.split()),
            "error_count": error_count,
            "status": "error",
        }


def _run_dialect_test(model: str, test_item: dict, temperature: float) -> dict:
    """Test model behavior with dialect/colloquial German."""
    input_text = test_item["corrupted"]
    expected = test_item.get("original", input_text)
    category = test_item.get("category", "unknown")
    note = test_item.get("note", "")

    try:
        response: ChatResponse = chat(
            model=model,
            messages=[
                {"role": "user", "content": input_text},
            ],
        )
        model_output = response["message"]["content"].strip()

        is_correct = _check_correction_quality(expected, model_output)

        return {
            "category": category,
            "note": note,
            "input_text": input_text,
            "expected_output": expected,
            "model_output": model_output,
            "was_modified": True,
            "is_correct": is_correct,
            "word_count": len(input_text.split()),
            "error_count": 1,
            "status": "success",
        }
    except Exception as e:
        return {
            "category": category,
            "note": note,
            "input_text": input_text,
            "expected_output": expected,
            "model_output": f"ERROR: {str(e)}",
            "was_modified": False,
            "is_correct": False,
            "word_count": len(input_text.split()),
            "error_count": 1,
            "status": "error",
        }


def _count_errors_in_text(corrupted: str, corrected: str) -> int:
    """Count approximate number of errors in the text."""
    import difflib

    matcher = difflib.SequenceMatcher(None, corrupted.lower(), corrected.lower())
    changes = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "replace":
            changes += max(i2 - i1, j2 - j1)
        elif tag == "insert":
            changes += j2 - j1
        elif tag == "delete":
            changes += i2 - i1
    return max(changes, 1)


def _check_correction_quality(expected: str, actual: str) -> bool:
    """Check if the correction is correct (exact match or very close)."""
    import difflib

    if expected.strip() == actual.strip():
        return True
    matcher = difflib.SequenceMatcher(None, expected.lower(), actual.lower())
    return matcher.ratio() > 0.80


def _compute_stress_summary(results: dict) -> None:
    """Compute summary statistics from stress test results."""
    total_tests = (
        len(results["length_tests"])
        + len(results["high_density_tests"])
        + len(results["dialect_tests"])
    )
    passed_tests = (
        sum(1 for r in results["length_tests"] if r["is_correct"])
        + sum(1 for r in results["high_density_tests"] if r["is_correct"])
        + sum(1 for r in results["dialect_tests"] if r["is_correct"])
    )

    results["summary"] = {
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "pass_rate": round(passed_tests / total_tests * 100, 1)
        if total_tests > 0
        else 0,
        "length_breakdown": {
            "short": _get_category_pass_rate(
                results["length_tests"], "very_short", "short"
            ),
            "medium": _get_category_pass_rate(results["length_tests"], "medium"),
            "long": _get_category_pass_rate(results["length_tests"], "long"),
            "very_long": _get_category_pass_rate(results["length_tests"], "very_long"),
        },
        "max_word_count_without_degradation": _find_max_word_count(
            results["length_tests"]
        ),
    }


def _get_category_pass_rate(tests: list, *categories) -> float:
    """Get pass rate for specific categories."""
    filtered = [t for t in tests if t.get("category") in categories]
    if not filtered:
        return 0.0
    return sum(1 for t in filtered if t["is_correct"]) / len(filtered) * 100


def _find_max_word_count(length_tests: list) -> int:
    """Find the maximum word count where the model still performs correctly."""
    max_correct = 0
    for test in length_tests:
        if test["is_correct"]:
            max_correct = max(max_correct, test.get("word_count", 0))
    return max_correct


def save_stress_results(results: dict, output_file: str) -> None:
    """Save stress test results to a JSON file."""
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
