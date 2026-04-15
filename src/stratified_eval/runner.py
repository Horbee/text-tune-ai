"""
Stratified Evaluation Runner.

Orchestrates the full stratified evaluation pipeline:
1. Tiered Evaluation (difficulty-based)
2. Over-Correction Benchmark (FPR)
3. Stress Testing
4. Error Type Analysis
5. Capabilities Matrix Generation
"""

import sys
import os
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from datetime import datetime

from ollama import Client

from stratified_eval.evaluation_data import (
    get_all_tier_data,
    get_flawless_data,
    get_stress_test_data,
)
from stratified_eval.tiered_evaluator import evaluate_all_tiers, save_tier_results
from stratified_eval.overcorrection_benchmark import run_fpr_benchmark
from stratified_eval.stress_tester import run_stress_tests, save_stress_results
from stratified_eval.capabilities_matrix import (
    generate_capabilities_matrix,
    save_capabilities_report,
)


def run_full_evaluation(
    model: str,
    judge_model: str,
    output_dir: str,
    run_tier_eval: bool = True,
    run_fpr: bool = True,
    run_stress: bool = True,
    run_matrix: bool = True,
):
    """
    Run the complete stratified evaluation pipeline.

    Args:
        model: The GEC model to evaluate
        judge_model: Model to use for LLM-based evaluation
        output_dir: Directory to save all results
        run_*: Flags to control which evaluations to run
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_safe = model.replace(":", "-")

    client = Client(
        host="https://ollama.com",
        headers={"Authorization": "Bearer " + os.environ.get("OLLAMA_API_KEY", "")},
    )

    print(f"\n{'#' * 60}")
    print(f"# STRATIFIED EVALUATION PIPELINE")
    print(f"# Model: {model}")
    print(f"# Timestamp: {timestamp}")
    print(f"{'#' * 60}")

    results = {"model": model, "timestamp": timestamp}
    tier_results = None
    fpr_results = None
    stress_results = None

    if run_tier_eval:
        print(f"\n{'=' * 60}")
        print("TIERED EVALUATION")
        print(f"{'=' * 60}")
        tier_data = get_all_tier_data()
        tier_results = evaluate_all_tiers(model, tier_data, client, judge_model)
        tier_file = f"{output_dir}/{model_safe}_tier_results_{timestamp}.json"
        save_tier_results(tier_results, tier_file)
        results["tier_results"] = tier_file
        print(f"\nTier Results Summary:")
        print(f"  Tier 1 (Easy):   {tier_results['summary']['tier_1_accuracy']}%")
        print(f"  Tier 2 (Medium): {tier_results['summary']['tier_2_accuracy']}%")
        print(f"  Tier 3 (Hard):   {tier_results['summary']['tier_3_accuracy']}%")
        print(f"  Overall:         {tier_results['summary']['overall_accuracy']}%")

    if run_fpr:
        print(f"\n{'=' * 60}")
        print("OVER-CORRECTION BENCHMARK (FPR)")
        print(f"{'=' * 60}")
        flawless_data = get_flawless_data()
        fpr_results = run_fpr_benchmark(model, flawless_data)
        fpr_file = f"{output_dir}/{model_safe}_fpr_results_{timestamp}.json"
        with open(fpr_file, "w", encoding="utf-8") as f:
            json.dump(fpr_results, f, ensure_ascii=False, indent=2)
        results["fpr_results"] = fpr_file
        print(f"\nFPR: {fpr_results['fpr_percentage']}%")
        print(f"Safety: {fpr_results['safety_rating']}")

    if run_stress:
        print(f"\n{'=' * 60}")
        print("STRESS TESTING")
        print(f"{'=' * 60}")
        stress_data = get_stress_test_data()
        stress_results = run_stress_tests(model, stress_data)
        stress_file = f"{output_dir}/{model_safe}_stress_results_{timestamp}.json"
        save_stress_results(stress_results, stress_file)
        results["stress_results"] = stress_file
        print(f"\nStress Test Pass Rate: {stress_results['summary']['pass_rate']}%")
        print(
            f"Max Word Count: {stress_results['summary']['max_word_count_without_degradation']}"
        )

    if run_matrix:
        print(f"\n{'=' * 60}")
        print("CAPABILITIES MATRIX")
        print(f"{'=' * 60}")

        tier_acc = {}
        if run_tier_eval and tier_results is not None:
            tier_acc = {
                "tier_1": tier_results.get("summary", {}).get("tier_1_accuracy", 0),
                "tier_2": tier_results.get("summary", {}).get("tier_2_accuracy", 0),
                "tier_3": tier_results.get("summary", {}).get("tier_3_accuracy", 0),
            }

        model_size = _infer_model_size(model)

        capabilities = generate_capabilities_matrix(
            model_name=model,
            tier_results=tier_acc,
            fpr_results=fpr_results
            if fpr_results is not None
            else {"fpr_percentage": 0},
            stress_results=stress_results
            if stress_results is not None
            else {"summary": {}},
            error_matrix={},
            model_size=model_size,
        )

        json_file = f"{output_dir}/{model_safe}_capabilities_{timestamp}.json"
        md_file = f"{output_dir}/{model_safe}_capabilities_{timestamp}.md"

        save_capabilities_report(capabilities, json_file, "json")
        save_capabilities_report(capabilities, md_file, "markdown")

        results["capabilities_json"] = json_file
        results["capabilities_md"] = md_file

        print(f"\nCapabilities Matrix saved to:")
        print(f"  {json_file}")
        print(f"  {md_file}")

    summary_file = f"{output_dir}/{model_safe}_summary_{timestamp}.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print(f"PIPELINE COMPLETE")
    print(f"{'=' * 60}")
    print(f"Summary saved to: {summary_file}")

    return results


def _infer_model_size(model_name: str) -> str:
    """Infer model size from model name."""
    name_lower = model_name.lower()
    if "small" in name_lower:
        return "3B"
    elif "base" in name_lower:
        return "8B"
    elif "large" in name_lower:
        return "13B"
    elif "1b" in name_lower:
        return "1B"
    elif "2b" in name_lower:
        return "2B"
    elif "3b" in name_lower:
        return "3B"
    elif "8b" in name_lower:
        return "8B"
    elif "11b" in name_lower:
        return "11B"
    elif "14b" in name_lower:
        return "14B"
    elif "70b" in name_lower or "72b" in name_lower:
        return "70B"
    return "unknown"


def main():
    parser = argparse.ArgumentParser(
        description="Stratified Evaluation Pipeline for German GEC Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python runner.py --model ministral-3:8b --judge-model gpt-oss:120b-cloud
  python runner.py --model phi3:3b --output-dir ./results --skip-fpr
  python runner.py --model mistral:7b --only-tier-eval
        """,
    )

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        help="GEC model to evaluate (e.g., ministral-3:8b)",
    )
    parser.add_argument(
        "--judge-model",
        "-j",
        type=str,
        default="gpt-oss:120b-cloud",
        help="LLM judge model (default: gpt-oss:120b-cloud)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="./stratified_eval_results",
        help="Output directory for results (default: ./stratified_eval_results)",
    )
    parser.add_argument(
        "--skip-tier-eval", action="store_true", help="Skip tiered evaluation"
    )
    parser.add_argument(
        "--skip-fpr", action="store_true", help="Skip over-correction (FPR) benchmark"
    )
    parser.add_argument(
        "--skip-stress", action="store_true", help="Skip stress testing"
    )
    parser.add_argument(
        "--only-matrix",
        action="store_true",
        help="Only generate capabilities matrix from existing results",
    )

    args = parser.parse_args()

    if args.only_matrix:
        print(
            "Error: --only-matrix requires existing result files (not yet implemented)"
        )
        return

    run_full_evaluation(
        model=args.model,
        judge_model=args.judge_model,
        output_dir=args.output_dir,
        run_tier_eval=not args.skip_tier_eval,
        run_fpr=not args.skip_fpr,
        run_stress=not args.skip_stress,
        run_matrix=True,
    )


if __name__ == "__main__":
    main()
