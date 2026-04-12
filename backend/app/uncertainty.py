"""Layer 3 — Multi-signal uncertainty detection.

OWNED BY PERSON A. See HANDOFF_PERSON_A.md for the full spec.
"""
from __future__ import annotations

from .schemas import ScribeWord, WordWithConfidence


def score_words(
    words: list[ScribeWord],
    keyterms: list[str],
    phonetic_map: dict[str, str],
    correction_history: dict[str, int],
) -> list[WordWithConfidence]:
    """Compute composite confidence per word from four signals.

    Signals (combine into a single score 0..1):
        1. Timing irregularity — duration vs rolling 5-word median; z-score > 1.5 → +0.3
        2. Keyterm mismatch     — word matches medical_patterns.matches_medical()
                                  but is NOT in the keyterm set → +0.25
        3. Phonetic distance    — min Levenshtein to keyterms; dist 1 or 2 → +0.3
        4. Correction likelihood — word appears in correction_history → +0.15

    Buckets: <0.25 HIGH, 0.25–0.5 MEDIUM, ≥0.5 LOW

    For LOW/MEDIUM words, populate uncertainty_signals with the names of the
    contributing signals so the frontend tooltip can render them, e.g.
        ["timing_irregularity", "phonetic_distance: 1", "keyterm_mismatch"]

    Return a list parallel to `words` (same length, same order, same speaker_id).
    """
    raise NotImplementedError("uncertainty.score_words — Person A: see HANDOFF_PERSON_A.md")
