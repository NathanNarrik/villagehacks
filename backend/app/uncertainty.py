"""Layer 3 — Multi-signal uncertainty detection.

OWNED BY PERSON A. See HANDOFF_PERSON_A.md for the full spec.
"""
from __future__ import annotations

import logging
from statistics import median
from threading import Lock
from typing import Any

from .config import settings
from .medical_patterns import matches_medical, normalize
from .phonetic import normalized_levenshtein
from .schemas import ScribeWord, WordWithConfidence

log = logging.getLogger(__name__)


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


def _bucket(score: float) -> str:
    if score < 0.25:
        return "HIGH"
    if score < 0.5:
        return "MEDIUM"
    return "LOW"


class _XGBoostRiskScorer:
    """Optional runtime scorer. Falls back silently if model/deps are unavailable."""

    def __init__(self) -> None:
        self._loaded = False
        self._model: Any | None = None
        self._pd: Any | None = None
        self._feature_names: list[str] = []
        self._lock = Lock()

    def _ensure_loaded(self) -> bool:
        with self._lock:
            if self._loaded:
                return self._model is not None and self._pd is not None

            self._loaded = True
            try:
                import joblib  # type: ignore[import-not-found]
                import pandas as pd  # type: ignore[import-not-found]
            except Exception:
                return False

            if not settings.XGBOOST_MODEL_PATH.exists():
                return False

            try:
                self._model = joblib.load(settings.XGBOOST_MODEL_PATH)
                self._pd = pd
                self._feature_names = list(
                    getattr(self._model, "feature_names_in_", []) or []
                )
                log.info(
                    "Loaded optional XGBoost uncertainty model from %s",
                    settings.XGBOOST_MODEL_PATH,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("Failed loading XGBoost model: %s", exc)
                self._model = None
                self._pd = None
                self._feature_names = []

            return self._model is not None and self._pd is not None

    def risk_for_word(self, features: dict[str, Any]) -> float | None:
        if not self._ensure_loaded():
            return None

        assert self._model is not None
        assert self._pd is not None

        base = {
            "voice_type": "unknown",
            "speech_style": "unknown",
            "accent": "unknown",
            "noise_level": "unknown",
            "scenario": "unknown",
            "has_interruptions": 0,
            "contains_ambiguity": 0,
            "contains_medical_terms": 0,
            "duration_sec_file": 0.0,
            "rms_mean": 0.0,
            "rms_std": 0.0,
            "zcr_mean": 0.0,
            "zcr_std": 0.0,
            "centroid_mean": 0.0,
            "centroid_std": 0.0,
            "rolloff_mean": 0.0,
            "rolloff_std": 0.0,
        }
        for i in range(1, 14):
            base[f"mfcc_{i}_mean"] = 0.0
            base[f"mfcc_{i}_std"] = 0.0
        base.update(features)

        # If model remembers feature names, align exactly to reduce inference drift.
        if self._feature_names:
            row = {name: base.get(name, 0.0) for name in self._feature_names}
        else:
            row = base

        try:
            df = self._pd.DataFrame([row])
            if hasattr(self._model, "predict_proba"):
                probs = self._model.predict_proba(df)[0]
                # Generic risk proxy for non-binary models: lower confidence => higher risk.
                max_prob = max(float(p) for p in probs) if len(probs) else 1.0
                return max(0.0, min(1.0, 1.0 - max_prob))
        except Exception as exc:  # noqa: BLE001
            log.warning("XGBoost inference failed; falling back to rule-based: %s", exc)
        return None


_xgb_scorer = _XGBoostRiskScorer()


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
    if not words:
        return []

    keyterms_norm = {normalize(k) for k in keyterms if k}
    durations = [max(0, w.end_ms - w.start_ms) for w in words]

    out: list[WordWithConfidence] = []
    for i, w in enumerate(words):
        word_norm = normalize(w.text)
        score = 0.0
        signals: list[str] = []

        # 1) Timing irregularity (rolling 5-word median with z-style check)
        left = max(0, i - 2)
        right = min(len(words), i + 3)
        window = durations[left:right]
        med = float(median(window)) if window else 0.0
        if window:
            mean = sum(window) / len(window)
            variance = sum((d - mean) ** 2 for d in window) / max(1, len(window))
            std = variance ** 0.5
            z_like = abs((durations[i] - med) / std) if std > 1e-6 else 0.0
            if z_like > 1.5:
                score += 0.30
                signals.append("timing_irregularity")

        # 2) Keyterm mismatch
        if word_norm and matches_medical(word_norm) and word_norm not in keyterms_norm:
            score += 0.25
            signals.append("keyterm_mismatch")

        # 3) Phonetic distance to nearest keyterm
        nearest_dist: int | None = None
        if word_norm and keyterms_norm:
            # Small optimization: use normalized distance to shortlist.
            shortlist: list[tuple[str, float]] = []
            for kt in keyterms_norm:
                shortlist.append((kt, normalized_levenshtein(word_norm, kt)))
            shortlist.sort(key=lambda x: x[1])
            for kt, _ in shortlist[:12]:
                dist = _levenshtein(word_norm, kt)
                if nearest_dist is None or dist < nearest_dist:
                    nearest_dist = dist
                if dist == 0:
                    break

        if nearest_dist in (1, 2):
            score += 0.30
            signals.append(f"phonetic_distance: {nearest_dist}")

        # 4) Correction likelihood from prior calls
        history_hits = correction_history.get(word_norm, 0)
        if history_hits > 0 or word_norm in phonetic_map:
            score += 0.15
            signals.append("correction_likelihood")

        # Optional Phase 2: XGBoost risk scoring.
        xgb_risk = _xgb_scorer.risk_for_word(
            {
                "duration_sec_file": max(0.0, (w.end_ms - w.start_ms) / 1000.0),
                "contains_ambiguity": int(bool(nearest_dist and nearest_dist <= 2)),
                "contains_medical_terms": int(matches_medical(word_norm)),
                "has_interruptions": int("..." in w.text or "--" in w.text),
            }
        )
        if xgb_risk is not None:
            # Use the more conservative signal (max risk) to reduce silent failures.
            if xgb_risk >= settings.XGBOOST_LOW_THRESHOLD:
                score = max(score, settings.XGBOOST_LOW_THRESHOLD)
            elif xgb_risk >= settings.XGBOOST_MEDIUM_THRESHOLD:
                score = max(score, settings.XGBOOST_MEDIUM_THRESHOLD)
            if xgb_risk >= settings.XGBOOST_MEDIUM_THRESHOLD:
                signals.append(f"xgboost_risk: {xgb_risk:.2f}")

        score = max(0.0, min(1.0, score))
        confidence = _bucket(score)

        out.append(
            WordWithConfidence(
                word=w.text,
                start_ms=w.start_ms,
                end_ms=w.end_ms,
                speaker_id=w.speaker_id,
                confidence=confidence,  # type: ignore[arg-type]
                uncertainty_signals=signals if confidence in ("LOW", "MEDIUM") else [],
            )
        )

    return out
