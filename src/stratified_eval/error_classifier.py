"""
Automatic error type classifier for GEC analysis.
Classifies errors into categories for granular error typology tracking.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional


class ErrorType(Enum):
    ORTHOGRAPHY = "orthography"
    CAPITALIZATION = "capitalization"
    PUNCTUATION = "punctuation"
    VERB_CONJUGATION = "verb_conjugation"
    VERB_AGREEMENT = "verb_agreement"
    SUBJECT_VERB_AGREEMENT = "subject-verb_agreement"
    ADJECTIVE_ENDING = "adjective_ending"
    ARTICLE_CASE = "article_case"
    PREPOSITION_CASE = "preposition_case"
    POSSESSIVE_PRONOUN_CASE = "possessive_pronoun_case"
    RELATIVE_PRONOUN_CASE = "relative_pronoun_case"
    RELATIVE_CLAUSE_COMMA = "relative_clause_comma"
    WORD_ORDER = "word_order"
    TENSE = "tense"
    SUBJUNCTIVE = "subjunctive"
    PASSIVE_VOICE = "passive_voice"
    INFINITIVE_CONSTRUCTION = "infinitive_construction"
    LANGUAGE_MIXING = "language_mixing"
    NESTED_CLAUSES = "nested_clauses"
    OTHER = "other"


@dataclass
class ErrorClassification:
    error_type: ErrorType
    confidence: float
    location: str
    original_segment: str
    corrected_segment: str


def classify_error_type(original: str, corrected: str) -> list[ErrorClassification]:
    """
    Classify the type of error between original and corrected text.
    This is a rule-based classifier for German GEC errors.
    """
    classifications = []
    orig_lower = original.lower()
    corr_lower = corrected.lower()

    if _has_capitalization_change(original, corrected):
        classifications.append(
            ErrorClassification(
                error_type=ErrorType.CAPITALIZATION,
                confidence=0.95,
                location="noun",
                original_segment=_find_changed_segment(original, corrected),
                corrected_segment="",
            )
        )

    if _has_punctuation_change(original, corrected):
        classifications.append(
            ErrorClassification(
                error_type=ErrorType.PUNCTUATION,
                confidence=0.9,
                location="sentence",
                original_segment=_find_changed_segment(original, corrected),
                corrected_segment="",
            )
        )

    if _has_orthography_change(original, corrected):
        classifications.append(
            ErrorClassification(
                error_type=ErrorType.ORTHOGRAPHY,
                confidence=0.85,
                location="word",
                original_segment=_find_changed_segment(original, corrected),
                corrected_segment="",
            )
        )

    if _has_verb_conjugation_change(original, corrected):
        classifications.append(
            ErrorClassification(
                error_type=ErrorType.VERB_CONJUGATION,
                confidence=0.8,
                location="verb",
                original_segment=_find_changed_segment(original, corrected),
                corrected_segment="",
            )
        )

    if _has_agreement_change(original, corrected):
        classifications.append(
            ErrorClassification(
                error_type=ErrorType.SUBJECT_VERB_AGREEMENT,
                confidence=0.75,
                location="subject-verb",
                original_segment=_find_changed_segment(original, corrected),
                corrected_segment="",
            )
        )

    if _has_adjective_ending_change(original, corrected):
        classifications.append(
            ErrorClassification(
                error_type=ErrorType.ADJECTIVE_ENDING,
                confidence=0.8,
                location="adjective",
                original_segment=_find_changed_segment(original, corrected),
                corrected_segment="",
            )
        )

    if _has_case_change(original, corrected):
        classifications.append(
            ErrorClassification(
                error_type=ErrorType.ARTICLE_CASE,
                confidence=0.7,
                location="article-noun",
                original_segment=_find_changed_segment(original, corrected),
                corrected_segment="",
            )
        )

    if _has_word_order_change(original, corrected):
        classifications.append(
            ErrorClassification(
                error_type=ErrorType.WORD_ORDER,
                confidence=0.75,
                location="sentence",
                original_segment=_find_changed_segment(original, corrected),
                corrected_segment="",
            )
        )

    if _has_language_mixing(original, corrected):
        classifications.append(
            ErrorClassification(
                error_type=ErrorType.LANGUAGE_MIXING,
                confidence=0.9,
                location="sentence",
                original_segment=_find_changed_segment(original, corrected),
                corrected_segment="",
            )
        )

    if not classifications:
        classifications.append(
            ErrorClassification(
                error_type=ErrorType.OTHER,
                confidence=0.5,
                location="unknown",
                original_segment="",
                corrected_segment="",
            )
        )

    return classifications


def _has_capitalization_change(original: str, corrected: str) -> bool:
    """Check if there's a capitalization change."""
    import re

    orig_words = original.split()
    corr_words = corrected.split()
    if len(orig_words) != len(corr_words):
        return False
    for o, c in zip(orig_words, corr_words):
        if o.lower() == c.lower() and o != c:
            return True
        if re.match(r"^[A-ZÄÖÜ][a-zäöüß]+$", o) and o.lower() == c.lower():
            return True
    return False


def _has_punctuation_change(original: str, corrected: str) -> bool:
    """Check if there's a punctuation change."""
    import re

    orig_punct = set(re.findall(r"[.,;:!?]", original))
    corr_punct = set(re.findall(r"[.,;:!?]", corrected))
    return orig_punct != corr_punct


def _has_orthography_change(original: str, corrected: str) -> bool:
    """Check if there's an orthography/spelling change."""
    orig_lower = original.lower()
    corr_lower = corrected.lower()
    if orig_lower == corr_lower:
        return False
    common_chars = set("äöüß")
    for char in common_chars:
        if char in orig_lower or char in corr_lower:
            if orig_lower.replace(char, "") != corr_lower.replace(char, ""):
                return True
    return False


def _has_verb_conjugation_change(original: str, corrected: str) -> bool:
    """Heuristic for verb conjugation changes."""
    verb_endings = ["st", "en", "e", "t", "est", "et", "te", "ten", "tet"]
    orig_words = original.lower().split()
    corr_words = corrected.lower().split()
    for i, (o, c) in enumerate(zip(orig_words, corr_words)):
        for ending in verb_endings:
            if o.endswith(ending) and c.endswith(ending) and o != c:
                return True
    return False


def _has_agreement_change(original: str, corrected: str) -> bool:
    """Heuristic for subject-verb agreement changes."""
    import re

    orig_verbs = re.findall(r"\b(\w+)\b", original.lower())
    corr_verbs = re.findall(r"\b(\w+)\b", corrected.lower())
    if len(orig_verbs) != len(corr_verbs):
        return True
    return False


def _has_adjective_ending_change(original: str, corrected: str) -> bool:
    """Heuristic for adjective ending changes."""
    orig_lower = original.lower()
    corr_lower = corrected.lower()
    adjective_patterns = ["-er", "-e", "-en", "-em", "-es", "-en"]
    for pattern in adjective_patterns:
        if pattern in orig_lower and pattern in corr_lower:
            return True
    return False


def _has_case_change(original: str, corrected: str) -> bool:
    """Heuristic for case changes (nominative, accusative, dative, genitive)."""
    case_articles = {
        "nominative": ["der", "die", "das", "ein", "eine"],
        "accusative": ["den", "die", "das", "einen", "eine"],
        "dative": ["dem", "der", "den", "einem", "einer"],
        "genitive": ["des", "der", "dessen", "eines", "einer"],
    }
    orig_lower = original.lower()
    corr_lower = corrected.lower()
    for case_name, articles in case_articles.items():
        for art in articles:
            if art in orig_lower and art in corr_lower:
                return True
    return False


def _has_word_order_change(original: str, corrected: str) -> bool:
    """Heuristic for word order changes."""
    orig_words = original.lower().split()
    corr_words = corrected.lower().split()
    if orig_words[:3] != corr_words[:3]:
        return True
    if len(orig_words) > 5:
        mid_orig = orig_words[2:-2]
        mid_corr = corr_words[2:-2]
        if mid_orig != mid_corr:
            return True
    return False


def _has_language_mixing(original: str, corrected: str) -> bool:
    """Check for mixing of German with other languages."""
    english_words = [
        "the",
        "is",
        "was",
        "have",
        "has",
        "been",
        "being",
        "tomorrow",
        "yesterday",
    ]
    orig_lower = original.lower()
    corr_lower = corrected.lower()
    for word in english_words:
        if word in orig_lower or word in corr_lower:
            return True
    return False


def _find_changed_segment(original: str, corrected: str) -> str:
    """Find the segment that changed between original and corrected."""
    orig_words = original.split()
    corr_words = corrected.split()
    changed = []
    for i, (o, c) in enumerate(zip(orig_words, corr_words)):
        if o.lower() != c.lower():
            changed.append(o)
    return " ".join(changed) if changed else ""


def compute_error_matrix(results: list[dict]) -> dict:
    """
    Compute an error matrix from evaluation results.
    Returns counts per error type per model/tier.
    """
    error_matrix = {}
    for result in results:
        model = result.get("model", "unknown")
        tier = result.get("tier", 0)
        if model not in error_matrix:
            error_matrix[model] = {}
        if tier not in error_matrix[model]:
            error_matrix[model][tier] = {}

        errors = classify_error_type(
            result.get("original", ""), result.get("corrected", "")
        )
        for err in errors:
            err_name = err.error_type.value
            if err_name not in error_matrix[model][tier]:
                error_matrix[model][tier][err_name] = {"total": 0, "corrected": 0}
            error_matrix[model][tier][err_name]["total"] += 1
            if result.get("is_correct", False):
                error_matrix[model][tier][err_name]["corrected"] += 1

    return error_matrix
