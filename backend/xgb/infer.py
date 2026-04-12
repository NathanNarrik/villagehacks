"""Inference utilities for the word-risk XGBoost model."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from app.schemas import ScribeWord

from .features import (
    DEFAULT_XGB_MODEL_PATH,
    WordFeatureRow,
    build_word_rows_for_clip,
)


@dataclass(slots=True)
class WordRiskScore:
    """Risk score for one word."""

    clip_id: str
    word_index: int
    word: str
    start_ms: int
    end_ms: int
    risk: float


@dataclass(slots=True)
class HighRiskSpan:
    """One grouped high-risk span."""

    clip_id: str
    token_start: int
    token_end: int
    text: str
    start_ms: int
    end_ms: int
    max_risk: float
    mean_risk: float


@dataclass(slots=True)
class InferenceResult:
    """Structured inference output."""

    word_scores: list[WordRiskScore]
    high_risk_spans: list[HighRiskSpan]


_CACHE_LOCK = threading.Lock()
_CACHE: dict[str, Any] = {
    "path": None,
    "mtime": None,
    "bundle": None,
}


def _load_bundle(path: Path = DEFAULT_XGB_MODEL_PATH) -> dict[str, Any] | None:
    """Load and cache a trained bundle."""
    try:
        import joblib  # type: ignore[import-not-found]
    except Exception:
        return None

    if not path.exists():
        return None

    mtime = path.stat().st_mtime
    with _CACHE_LOCK:
        if _CACHE["path"] == str(path) and _CACHE["mtime"] == mtime:
            bundle = _CACHE["bundle"]
            return bundle if isinstance(bundle, dict) else None

        bundle = joblib.load(path)
        if not isinstance(bundle, dict):
            return None
        _CACHE["path"] = str(path)
        _CACHE["mtime"] = mtime
        _CACHE["bundle"] = bundle
        return bundle


def score_feature_rows(
    rows: Sequence[WordFeatureRow],
    *,
    model_path: Path = DEFAULT_XGB_MODEL_PATH,
) -> list[float] | None:
    """Score feature rows with the saved model bundle."""
    bundle = _load_bundle(model_path)
    if bundle is None:
        return None

    try:
        import pandas as pd  # type: ignore[import-not-found]
    except Exception:
        return None

    preprocessor = bundle.get("preprocessor")
    classifier = bundle.get("classifier")
    raw_feature_names = list(bundle.get("raw_feature_names") or [])
    if preprocessor is None or classifier is None or not raw_feature_names:
        return None

    frame = pd.DataFrame([{name: row.to_dict().get(name) for name in raw_feature_names} for row in rows])
    matrix = preprocessor.transform(frame)
    if hasattr(classifier, "predict_proba"):
        probabilities = classifier.predict_proba(matrix)
        return [float(prob[1]) for prob in probabilities]
    predictions = classifier.predict(matrix)
    return [float(pred) for pred in predictions]


def _should_merge(previous: WordRiskScore, current: WordRiskScore, gap_threshold_ms: int) -> bool:
    return current.word_index == previous.word_index + 1 and (
        current.start_ms - previous.end_ms <= gap_threshold_ms
    )


def group_high_risk_spans(
    scores: Sequence[WordRiskScore],
    *,
    threshold: float,
    gap_threshold_ms: int = 250,
) -> list[HighRiskSpan]:
    """Group adjacent high-risk words into spans."""
    high = [score for score in scores if score.risk >= threshold]
    if not high:
        return []

    spans: list[list[WordRiskScore]] = [[high[0]]]
    for score in high[1:]:
        current_span = spans[-1]
        if _should_merge(current_span[-1], score, gap_threshold_ms):
            current_span.append(score)
        else:
            spans.append([score])

    grouped: list[HighRiskSpan] = []
    for span in spans:
        grouped.append(
            HighRiskSpan(
                clip_id=span[0].clip_id,
                token_start=span[0].word_index,
                token_end=span[-1].word_index,
                text=" ".join(item.word for item in span),
                start_ms=span[0].start_ms,
                end_ms=span[-1].end_ms,
                max_risk=max(item.risk for item in span),
                mean_risk=sum(item.risk for item in span) / len(span),
            )
        )
    return grouped


def score_transcript_words(
    *,
    clip_id: str,
    words: Sequence[ScribeWord],
    clip_metadata: dict[str, Any] | None = None,
    keyterms: Iterable[str] | None = None,
    correction_frequency: dict[str, int] | None = None,
    model_path: Path = DEFAULT_XGB_MODEL_PATH,
    low_threshold: float = 0.6,
) -> InferenceResult | None:
    """Build features, score a transcript, and group risky spans."""
    rows = build_word_rows_for_clip(
        clip_id=clip_id,
        words=words,
        clip_metadata=clip_metadata,
        corrected_text=None,
        keyterms=keyterms,
        correction_frequency=correction_frequency,
    )
    probabilities = score_feature_rows(rows, model_path=model_path)
    if probabilities is None:
        return None

    word_scores = [
        WordRiskScore(
            clip_id=clip_id,
            word_index=row.word_index,
            word=words[idx].text,
            start_ms=row.start_ms,
            end_ms=row.end_ms,
            risk=float(probabilities[idx]),
        )
        for idx, row in enumerate(rows)
    ]
    return InferenceResult(
        word_scores=word_scores,
        high_risk_spans=group_high_risk_spans(word_scores, threshold=low_threshold),
    )
