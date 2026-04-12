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
    aa = (a or "").strip().lower()
    bb = (b or "").strip().lower()

    if aa == bb:
        return 0.0
    if not aa and not bb:
        return 0.0
    if not aa or not bb:
        return 1.0

    m, n = len(aa), len(bb)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        ca = aa[i - 1]
        for j in range(1, n + 1):
            cb = bb[j - 1]
            cost = 0 if ca == cb else 1
            curr[j] = min(
                prev[j] + 1,      # deletion
                curr[j - 1] + 1,  # insertion
                prev[j - 1] + cost,  # substitution
            )
        prev = curr

    return prev[n] / max(m, n)
