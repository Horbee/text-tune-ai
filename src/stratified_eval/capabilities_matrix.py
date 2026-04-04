"""
Model Capabilities Matrix Generator.

Compiles all evaluation findings into a comprehensive capabilities matrix
for each model size, defining operational boundaries and use cases.
"""

import json
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class ModelCapabilities:
    model_name: str
    model_size: str
    ideal_use_case: str
    known_limitations: list[str]
    safety_rating: str
    fpr_percentage: float
    tier_accuracy: dict
    stress_test_pass_rate: float
    max_word_count: int
    error_type_accuracy: dict
    hardware_profile: str
    latency_profile: str


def generate_capabilities_matrix(
    model_name: str,
    tier_results: dict,
    fpr_results: dict,
    stress_results: dict,
    error_matrix: dict,
    model_size: str = "unknown",
) -> ModelCapabilities:
    """
    Generate a capabilities matrix for a given model.

    Args:
        model_name: Name of the model (e.g., "ministral-3:8b")
        tier_results: Dict with tier -> accuracy mapping
        fpr_results: Dict with FPR results
        stress_results: Dict with stress test results
        error_matrix: Dict with error type accuracy
        model_size: Size category (e.g., "1B", "3B", "8B")

    Returns:
        ModelCapabilities object
    """
    tier_acc = _summarize_tier_accuracy(tier_results)
    stress_pass = stress_results.get("summary", {}).get("pass_rate", 0)
    max_wc = stress_results.get("summary", {}).get(
        "max_word_count_without_degradation", 0
    )
    fpr = fpr_results.get("fpr_percentage", 0)

    ideal_use = _determine_ideal_use_case(model_size, tier_acc, stress_pass, fpr)
    limitations = _determine_limitations(model_size, tier_acc, stress_pass, max_wc, fpr)
    safety = _determine_safety_rating(fpr, stress_pass)
    hw_profile, latency = _estimate_hardware_profile(model_size)

    return ModelCapabilities(
        model_name=model_name,
        model_size=model_size,
        ideal_use_case=ideal_use,
        known_limitations=limitations,
        safety_rating=safety,
        fpr_percentage=fpr,
        tier_accuracy=tier_acc,
        stress_test_pass_rate=stress_pass,
        max_word_count=max_wc,
        error_type_accuracy=error_matrix,
        hardware_profile=hw_profile,
        latency_profile=latency,
    )


def _summarize_tier_accuracy(tier_results: dict) -> dict:
    """Summarize accuracy per tier."""
    if isinstance(tier_results, dict):
        if not any(k.startswith("tier_") for k in tier_results.keys()):
            return {f"tier_{k}": v for k, v in tier_results.items()}
        return tier_results
    return tier_results


def _determine_ideal_use_case(
    model_size: str, tier_acc: dict, stress_pass: float, fpr: float
) -> str:
    """Determine ideal use case based on capabilities."""
    tier1_acc = tier_acc.get("tier_1", 0)
    tier2_acc = tier_acc.get("tier_2", 0)
    tier3_acc = tier_acc.get("tier_3", 0)

    if model_size in ("1B", "2B", "3B"):
        if tier1_acc > 90 and fpr < 10:
            return "Real-time as-you-type spellcheck for short sentences; quick typo and capitalization fixes"
        elif tier1_acc > 80:
            return "Lightweight spellchecking; suitable for mobile or low-resource environments"
        else:
            return "Not recommended for production without human oversight"

    elif model_size in ("8B", "11B"):
        if tier1_acc > 95 and tier2_acc > 85 and fpr < 5:
            return "General-purpose GEC for medium-length texts; handles most grammar and style issues"
        elif tier1_acc > 90 and tier2_acc > 75:
            return "Standard grammar correction for user-generated content"
        else:
            return "Use with caution; results may require review"

    elif model_size in ("14B", "70B"):
        if tier1_acc > 98 and tier2_acc > 95 and tier3_acc > 80 and fpr < 3:
            return "Premium GEC for professional writing; handles complex syntax and long texts"
        else:
            return "High-quality GEC for formal documents; recommended for publishing"

    return "General-purpose grammar correction"


def _determine_limitations(
    model_size: str, tier_acc: dict, stress_pass: float, max_word_count: int, fpr: float
) -> list[str]:
    """Determine known limitations."""
    limitations = []

    tier1_acc = tier_acc.get("tier_1", 0)
    tier2_acc = tier_acc.get("tier_2", 0)
    tier3_acc = tier_acc.get("tier_3", 0)

    if tier1_acc < 85:
        limitations.append("Struggles with basic spelling and capitalization errors")

    if tier2_acc < 65:
        limitations.append(
            "May fail on context-dependent grammar (case, gender agreement)"
        )

    if tier3_acc < 40:
        limitations.append(
            "Unreliable on complex restructuring, passive voice, nested clauses"
        )

    if max_word_count < 20:
        limitations.append(f"Fails on sentences longer than ~{max_word_count} words")

    if fpr > 15:
        limitations.append(
            "High risk of over-correcting grammatically correct sentences"
        )

    if stress_pass < 70:
        limitations.append(
            "Reduced performance on edge cases (dialect, slang, mixed languages)"
        )

    if not limitations:
        limitations.append("No significant limitations identified in standard testing")

    return limitations


def _determine_safety_rating(fpr: float, stress_pass: float) -> str:
    """Determine overall safety rating."""
    if fpr < 5 and stress_pass > 90:
        return "HIGH - Safe for autonomous deployment with minimal oversight"
    elif fpr < 15 and stress_pass > 75:
        return "MEDIUM - Safe with standard human review workflow"
    elif fpr < 30 and stress_pass > 60:
        return "LOW - Requires human review before final output"
    else:
        return "MINIMAL - Do not deploy autonomously; requires human oversight"


def _estimate_hardware_profile(model_size: str) -> tuple:
    """Estimate hardware requirements and latency."""
    profiles = {
        "1B": ("CPU/iGPU capable", "~60 tokens/sec on modern CPU"),
        "2B": ("CPU/iGPU capable", "~45 tokens/sec on modern CPU"),
        "3B": ("CPU with 8GB+ RAM or entry GPU", "~35 tokens/sec on CPU, ~120 on GPU"),
        "8B": ("GPU recommended (6GB+ VRAM)", "~25 tokens/sec on CPU, ~180 on GPU"),
        "11B": ("GPU required (8GB+ VRAM)", "~20 tokens/sec on CPU, ~150 on GPU"),
        "14B": ("GPU required (10GB+ VRAM)", "~15 tokens/sec on CPU, ~130 on GPU"),
        "70B": (
            "High-end GPU required (24GB+ VRAM)",
            "~8 tokens/sec on CPU, ~80 on GPU",
        ),
    }
    return profiles.get(model_size, ("Unknown", "Unknown performance"))


def generate_matrix_markdown(capabilities: ModelCapabilities) -> str:
    """Generate a markdown representation of the capabilities matrix."""
    md = f"""
# Model Capabilities Matrix: {capabilities.model_name}

## Overview
| Property | Value |
|----------|-------|
| **Model Size** | {capabilities.model_size} |
| **Safety Rating** | {capabilities.safety_rating} |
| **False Positive Rate** | {capabilities.fpr_percentage}% |
| **Stress Test Pass Rate** | {capabilities.stress_test_pass_rate}% |
| **Max Word Count** | {capabilities.max_word_count} words |

## Ideal Use Case
{capabilities.ideal_use_case}

## Known Limitations
"""
    for i, lim in enumerate(capabilities.known_limitations, 1):
        md += f"{i}. {lim}\n"

    md += f"""
## Tier Accuracy
| Tier | Accuracy |
|------|----------|
| Tier 1 (Easy) | {capabilities.tier_accuracy.get("tier_1", "N/A")}% |
| Tier 2 (Medium) | {capabilities.tier_accuracy.get("tier_2", "N/A")}% |
| Tier 3 (Hard) | {capabilities.tier_accuracy.get("tier_3", "N/A")}% |

## Error Type Accuracy
| Error Type | Accuracy |
|------------|----------|
"""
    for err_type, acc in capabilities.error_type_accuracy.items():
        md += f"| {err_type} | {acc}% |\n"

    md += f"""
## Hardware & Latency Profile
| Property | Value |
|----------|-------|
| **Hardware** | {capabilities.hardware_profile} |
| **Latency** | {capabilities.latency_profile} |

---
*Generated by Stratified Evaluation Pipeline*
"""
    return md


def save_capabilities_report(
    capabilities: ModelCapabilities, output_file: str, format: str = "json"
) -> None:
    """Save the capabilities report to a file."""
    if format == "json":
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(asdict(capabilities), f, ensure_ascii=False, indent=2)
    elif format == "markdown":
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(generate_matrix_markdown(capabilities))
    else:
        raise ValueError(f"Unknown format: {format}. Use 'json' or 'markdown'.")


def generate_comparison_table(all_capabilities: list[ModelCapabilities]) -> str:
    """Generate a comparison table for multiple models."""
    md = "# Model Comparison Matrix\n\n"
    md += "| Model | Size | Safety | FPR | T1 Acc | T2 Acc | T3 Acc | Max Words | Stress Pass |\n"
    md += "|-------|------|--------|-----|--------|--------|--------|-----------|------------|\n"

    for cap in all_capabilities:
        md += (
            f"| {cap.model_name} | {cap.model_size} | {cap.safety_rating.split(' - ')[0]} | "
            f"{cap.fpr_percentage}% | {cap.tier_accuracy.get('tier_1', 'N/A')} | "
            f"{cap.tier_accuracy.get('tier_2', 'N/A')} | {cap.tier_accuracy.get('tier_3', 'N/A')} | "
            f"{cap.max_word_count} | {cap.stress_test_pass_rate}% |\n"
        )

    return md
