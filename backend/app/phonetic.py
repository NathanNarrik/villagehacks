"""Phonetic distance helpers — used by uncertainty.score_words.

OWNED BY PERSON A. See HANDOFF_PERSON_A.md for the full spec.
"""
from __future__ import annotations


def normalized_levenshtein(a: str, b: str) -> float:
    """Edit distance from a to b, normalized to [0, 1] by max length.

    Person A may also implement Double Metaphone matching to catch sound-alikes
    that look very different on paper (e.g. "ph" vs "f"). The pipeline only
    requires this single function be present.
    """
    raise NotImplementedError("phonetic.normalized_levenshtein — Person A: see HANDOFF_PERSON_A.md")
