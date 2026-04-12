"""Tavily verifier — dedupe, cap, cache hit, canonical extraction."""
from __future__ import annotations

import asyncio

import pytest

from app.schemas import WordWithConfidence
from app.storage import InMemoryStore
from app.tavily_verify import TavilyVerifier


def _wc(word: str) -> WordWithConfidence:
    return WordWithConfidence(
        word=word,
        start_ms=0,
        end_ms=100,
        speaker_id="speaker_0",
        confidence="LOW",
    )


class FakeTavilyClient:
    def __init__(self, results: list[dict]):
        self.results = results
        self.call_count = 0
        self.queries: list[str] = []

    def search(self, query: str, **kwargs) -> dict:
        self.call_count += 1
        self.queries.append(query)
        return {"results": self.results}


@pytest.fixture
def verifier_factory(monkeypatch):
    def _make(results: list[dict] | None = None, cap: int = 5):
        fake = FakeTavilyClient(results or [])
        v = TavilyVerifier(api_key="dummy", store=InMemoryStore(), cap=cap, ttl=3600)
        v._client = fake  # type: ignore[assignment]
        return v, fake

    return _make


@pytest.mark.asyncio
async def test_dedupe_collapses_repeats(verifier_factory):
    fake_results = [
        {"title": "Metformin (Glucophage)", "content": "metformin 500 mg", "url": "https://x"}
    ]
    verifier, fake = verifier_factory(results=fake_results)

    # Same word four times -> 1 Tavily call
    out = await verifier.verify_batch([_wc("metoformin"), _wc("metoformin"), _wc("metoformin")])
    assert fake.call_count == 1
    assert "metoformin" in out


@pytest.mark.asyncio
async def test_cap_limits_unique_calls(verifier_factory):
    fake_results = [{"title": "drug", "content": "drug", "url": "https://x"}]
    verifier, fake = verifier_factory(results=fake_results, cap=2)
    words = [_wc(w) for w in ("alpha", "bravo", "charlie", "delta")]
    await verifier.verify_batch(words)
    assert fake.call_count == 2


@pytest.mark.asyncio
async def test_cache_hit_on_second_call(verifier_factory):
    fake_results = [
        {"title": "Lisinopril", "content": "lisinopril 10 mg", "url": "https://x"}
    ]
    verifier, fake = verifier_factory(results=fake_results)

    await verifier.verify("lisinipril")
    await verifier.verify("lisinipril")
    assert fake.call_count == 1


@pytest.mark.asyncio
async def test_extract_canonical_picks_close_match(verifier_factory):
    fake_results = [
        {
            "title": "Atorvastatin (Lipitor)",
            "content": "atorvastatin is a statin used to lower cholesterol",
            "url": "https://example.com/atorvastatin",
        }
    ]
    verifier, _ = verifier_factory(results=fake_results)
    result = await verifier.verify("atorvastain")
    assert result.status == "VERIFIED"
    assert result.canonical == "atorvastatin"
    assert result.source_url == "https://example.com/atorvastatin"


@pytest.mark.asyncio
async def test_unverified_when_no_drug_token_in_results(verifier_factory):
    fake_results = [
        {"title": "weather forecast", "content": "sunny tomorrow", "url": "https://x"}
    ]
    verifier, _ = verifier_factory(results=fake_results)
    result = await verifier.verify("xyzabc")
    assert result.status == "UNVERIFIED"
    assert result.canonical is None
