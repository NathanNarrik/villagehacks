"""Layer 4 — Confidence-gated Tavily verification.

Only fires on LOW-confidence words that match the medical pattern gate. Capped at
TAVILY_CALL_CAP per transcript, deduped by normalized form, results cached in the
in-memory store with TTL.

Tavily is used for VALIDATION, not generation: we look up a candidate term and
return a clear pass/fail with a canonical replacement. Claude only corrects words
that come back VERIFIED.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict
from typing import Iterable

from tavily import TavilyClient

from .config import settings
from .medical_patterns import DRUG_SUFFIXES, normalize
from .schemas import VerifyResult, WordWithConfidence
from .storage import InMemoryStore, store as global_store

log = logging.getLogger(__name__)


def _levenshtein(a: str, b: str) -> int:
    """Plain DP edit distance — small inputs (drug names) so no optimization needed."""
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


class TavilyVerifier:
    """Confidence-gated Tavily lookup with dedupe, cap, and caching."""

    def __init__(
        self,
        api_key: str,
        store: InMemoryStore | None = None,
        cap: int = settings.TAVILY_CALL_CAP,
        ttl: int = settings.TAVILY_CACHE_TTL_SEC,
    ):
        self._client = TavilyClient(api_key=api_key) if api_key else None
        self._store = store or global_store
        self._cap = cap
        self._ttl = ttl

    async def verify(self, term: str) -> VerifyResult:
        """Verify a single term. Cached on success."""
        norm = normalize(term)
        if not norm:
            return VerifyResult(original=term, status="UNVERIFIED")

        cached = self._store.get(f"tavily:{norm}")
        if cached:
            return VerifyResult(**cached)

        if self._client is None:
            log.warning("Tavily client not configured — returning UNVERIFIED for %s", term)
            return VerifyResult(original=term, status="UNVERIFIED")

        try:
            res = await asyncio.to_thread(
                self._client.search,
                query=f"{norm} drug medication dosage",
                max_results=3,
                search_depth="basic",
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("Tavily search failed for %s: %s", term, exc)
            return VerifyResult(original=term, status="UNVERIFIED")

        canonical, source = self._extract_canonical(norm, res.get("results", []) or [])
        result = VerifyResult(
            original=term,
            status="VERIFIED" if canonical else "UNVERIFIED",
            canonical=canonical,
            source_url=source,
        )
        self._store.set(f"tavily:{norm}", asdict(result), ttl_sec=self._ttl)
        return result

    async def verify_batch(
        self, words: Iterable[WordWithConfidence]
    ) -> dict[str, VerifyResult]:
        """Dedupe → cap → run concurrently. Returns dict keyed by normalized term."""
        unique: list[str] = []
        seen: set[str] = set()
        for w in words:
            n = normalize(w.word)
            if n and n not in seen:
                seen.add(n)
                unique.append(n)
                if len(unique) >= self._cap:
                    break

        if not unique:
            return {}

        results = await asyncio.gather(*[self.verify(t) for t in unique])
        return {normalize(r.original): r for r in results}

    @staticmethod
    def _extract_canonical(
        query: str, results: list[dict]
    ) -> tuple[str | None, str | None]:
        """Walk Tavily result titles + snippets, find the closest drug-suffix token to query.

        A token is accepted as the canonical form if its normalized edit distance to
        `query` is below 0.4 — i.e. clearly the same word, just spelled correctly.
        """
        best_token: str | None = None
        best_dist = float("inf")
        best_url: str | None = None

        for item in results:
            text_blobs = [item.get("title") or "", item.get("content") or ""]
            url = item.get("url")
            for blob in text_blobs:
                for raw_tok in blob.split():
                    tok = normalize(raw_tok)
                    if len(tok) < 4 or not DRUG_SUFFIXES.match(tok):
                        continue
                    if tok == query:
                        return tok, url
                    dist = _levenshtein(tok, query)
                    norm_dist = dist / max(len(query), 1)
                    if norm_dist < 0.4 and dist < best_dist:
                        best_token = tok
                        best_dist = dist
                        best_url = url

        return best_token, best_url


# Module-level singleton — built lazily so tests can override settings first.
_verifier: TavilyVerifier | None = None


def get_verifier() -> TavilyVerifier:
    global _verifier
    if _verifier is None:
        _verifier = TavilyVerifier(api_key=settings.TAVILY_API_KEY.get_secret_value())
    return _verifier


def reset_verifier() -> None:
    """Test helper — drop the cached singleton so a fresh client is built."""
    global _verifier
    _verifier = None
