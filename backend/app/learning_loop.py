"""Layer 7 — Adaptive vocabulary memory.

After each successful pipeline run, the learning loop records:
- Verified corrections as a phonetic map (wrong_spelling -> canonical)
- Per-word correction frequency (feeds Layer 3's "correction likelihood" signal)
- Verified canonical terms in a sorted set keyed by frequency (becomes the keyterm
  list fed back into Scribe v2 on the next call)

All state lives in the in-process store from `app/storage.py`. State resets when
the process restarts — acceptable for hackathon demos where the "improves with each
call" pitch only needs within-process memory.
"""
from __future__ import annotations

from typing import Any

from .schemas import CorrectedWord, VerifyResult, WordWithConfidence
from .storage import store

KEYTERMS_KEY = "cc:keyterms"
PHONETIC_MAP_KEY = "cc:phonetic_map"
CORRECTION_HISTORY_KEY = "cc:correction_history"


def get_keyterms(top_n: int = 100) -> list[str]:
    """Return the most frequent learned keyterms, falling back to Person A's initial list."""
    learned = store.zrevrange(KEYTERMS_KEY, top_n)
    if learned:
        return learned
    # Person A owns this — wrapped in try so the backend boots without it.
    try:
        from . import keyterms as keyterms_module

        return keyterms_module.load_initial_keyterms()
    except (NotImplementedError, ImportError, AttributeError):
        return []


def get_phonetic_map() -> dict[str, str]:
    return {k: str(v) for k, v in store.hgetall(PHONETIC_MAP_KEY).items()}


def get_correction_history() -> dict[str, int]:
    return {k: int(v) for k, v in store.hgetall(CORRECTION_HISTORY_KEY).items()}


def keyterm_count() -> int:
    return store.zsize(KEYTERMS_KEY)


def phonetic_map_size() -> int:
    return len(store.hgetall(PHONETIC_MAP_KEY))


def record_call(
    raw_words: list[WordWithConfidence],
    corrected: list[CorrectedWord],
    verifications: dict[str, Any],
) -> None:
    """Persist what we learned from this transcript.

    `raw_words` is parallel to `corrected`, so index-aligned access gives us each
    correction's original spelling.
    """
    for idx, cw in enumerate(corrected):
        if cw.changed and cw.tavily_verified and idx < len(raw_words):
            original = raw_words[idx].word.lower().strip(",.;:!?")
            canonical = cw.word.lower().strip(",.;:!?")
            if original and canonical:
                store.hset(PHONETIC_MAP_KEY, original, canonical)
                store.hincrby(CORRECTION_HISTORY_KEY, original, 1)
                store.zincrby(KEYTERMS_KEY, canonical, 1.0)

    for v in verifications.values():
        if isinstance(v, VerifyResult) and v.status == "VERIFIED" and v.canonical:
            store.zincrby(KEYTERMS_KEY, v.canonical.lower(), 1.0)
