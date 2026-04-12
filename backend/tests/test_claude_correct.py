"""Hallucination guard — the safety-critical invariant.

The contract: any CorrectedWord with `changed=True` MUST have `tavily_verified=True`
AND the corrected text must appear as a canonical in the verifications dict. If
Claude hallucinates either condition, the entry is reverted to the original.
"""
from __future__ import annotations

import json

import pytest

from app.claude_correct import ClaudeCorrector
from app.schemas import VerifyResult, WordWithConfidence


def _wc(word: str, confidence="LOW") -> WordWithConfidence:
    return WordWithConfidence(
        word=word, start_ms=0, end_ms=100, speaker_id="speaker_0", confidence=confidence
    )


class FakeMessages:
    def __init__(self, payload: dict):
        self._payload = payload
        self.calls = 0

    async def create(self, **kwargs):
        self.calls += 1

        class _Resp:
            def __init__(self, txt):
                self.content = [type("Block", (), {"text": txt})()]

        return _Resp(json.dumps(self._payload))


class FakeAnthropic:
    def __init__(self, payload: dict):
        self.messages = FakeMessages(payload)


def _make_corrector(payload: dict) -> ClaudeCorrector:
    c = ClaudeCorrector(api_key="dummy")
    c._client = FakeAnthropic(payload)  # type: ignore[assignment]
    return c


@pytest.mark.asyncio
async def test_guard_reverts_unverified_change():
    """Claude tries to correct 'metoformin' -> 'metformin' but tavily_verified=false.
    Guard must revert to original."""
    raw = [_wc("metoformin")]
    verifications = {"metoformin": VerifyResult(original="metoformin", status="UNVERIFIED")}
    payload = {
        "corrections": [
            {"index": 0, "corrected": "metformin", "tavily_verified": False, "unverified": False}
        ]
    }
    corrector = _make_corrector(payload)
    result = await corrector.correct(raw, verifications, speakers=["Doctor"])
    assert result[0].word == "metoformin"  # reverted
    assert result[0].changed is False
    assert result[0].tavily_verified is False
    assert result[0].unverified is True


@pytest.mark.asyncio
async def test_guard_reverts_when_canonical_not_in_dict():
    """Claude claims tavily_verified=true but the canonical it produces was never
    actually verified by Tavily — must still revert."""
    raw = [_wc("metoformin")]
    verifications = {"metoformin": VerifyResult(original="metoformin", status="UNVERIFIED")}
    payload = {
        "corrections": [
            {"index": 0, "corrected": "metformin", "tavily_verified": True, "unverified": False}
        ]
    }
    corrector = _make_corrector(payload)
    result = await corrector.correct(raw, verifications, speakers=["Doctor"])
    assert result[0].word == "metoformin"
    assert result[0].changed is False
    assert result[0].tavily_verified is False


@pytest.mark.asyncio
async def test_guard_allows_verified_change():
    """Both flags pass: tavily_verified=true AND canonical matches the verifications dict."""
    raw = [_wc("metoformin")]
    verifications = {
        "metoformin": VerifyResult(
            original="metoformin",
            status="VERIFIED",
            canonical="metformin",
            source_url="https://x",
        )
    }
    payload = {
        "corrections": [
            {"index": 0, "corrected": "metformin", "tavily_verified": True, "unverified": False}
        ]
    }
    corrector = _make_corrector(payload)
    result = await corrector.correct(raw, verifications, speakers=["Doctor"])
    assert result[0].word == "metformin"
    assert result[0].changed is True
    assert result[0].tavily_verified is True


@pytest.mark.asyncio
async def test_unflagged_words_pass_through_unchanged():
    raw = [_wc("hello", confidence="HIGH"), _wc("metoformin", confidence="LOW")]
    verifications = {
        "metoformin": VerifyResult(
            original="metoformin", status="VERIFIED", canonical="metformin"
        )
    }
    payload = {
        "corrections": [
            {"index": 1, "corrected": "metformin", "tavily_verified": True, "unverified": False}
        ]
    }
    corrector = _make_corrector(payload)
    result = await corrector.correct(raw, verifications, speakers=["Patient", "Doctor"])
    assert result[0].word == "hello"
    assert result[0].speaker == "Patient"
    assert result[0].changed is False
    assert result[1].word == "metformin"
    assert result[1].changed is True
