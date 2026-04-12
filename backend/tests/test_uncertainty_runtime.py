from __future__ import annotations

from app import uncertainty
from app.schemas import ScribeWord


def _w(text: str, start_ms: int, end_ms: int) -> ScribeWord:
    return ScribeWord(text=text, start_ms=start_ms, end_ms=end_ms, speaker_id="speaker_0")


def test_uncertainty_rule_based_without_xgboost(monkeypatch):
    monkeypatch.setattr(uncertainty._xgb_scorer, "risk_for_word", lambda features: None)

    words = [
        _w("metoformin", 0, 120),
        _w("500mg", 120, 260),
        _w("daily", 260, 360),
    ]
    scored = uncertainty.score_words(
        words=words,
        keyterms=["metformin"],
        phonetic_map={},
        correction_history={},
    )

    assert len(scored) == 3
    assert scored[0].confidence in ("LOW", "MEDIUM")
    assert any("phonetic_distance" in s for s in scored[0].uncertainty_signals)


def test_uncertainty_uses_xgboost_risk_when_available(monkeypatch):
    monkeypatch.setattr(uncertainty._xgb_scorer, "risk_for_word", lambda features: 0.81)

    words = [_w("hello", 0, 80)]
    scored = uncertainty.score_words(
        words=words,
        keyterms=[],
        phonetic_map={},
        correction_history={},
    )

    assert scored[0].confidence == "LOW"
    assert any(sig.startswith("xgboost_risk:") for sig in scored[0].uncertainty_signals)
