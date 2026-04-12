"""In-process key/value store with TTL support.

Mimics a small subset of Redis (string get/set, hash, sorted set) so that swapping
in real Redis later requires only changing this file. No locking — FastAPI runs in a
single asyncio event loop per worker so all access is serialized.
"""
from __future__ import annotations

import time
from typing import Any


class InMemoryStore:
    def __init__(self) -> None:
        self._strings: dict[str, Any] = {}
        self._exp: dict[str, float] = {}
        self._hashes: dict[str, dict[str, Any]] = {}
        self._zsets: dict[str, dict[str, float]] = {}

    # --- string ops with optional TTL ---
    def get(self, key: str) -> Any | None:
        if self._is_expired(key):
            self._strings.pop(key, None)
            self._exp.pop(key, None)
            return None
        return self._strings.get(key)

    def set(self, key: str, value: Any, ttl_sec: int | None = None) -> None:
        self._strings[key] = value
        if ttl_sec is not None:
            self._exp[key] = time.time() + ttl_sec
        else:
            self._exp.pop(key, None)

    def delete(self, key: str) -> None:
        self._strings.pop(key, None)
        self._exp.pop(key, None)
        self._hashes.pop(key, None)
        self._zsets.pop(key, None)

    def _is_expired(self, key: str) -> bool:
        exp = self._exp.get(key)
        return exp is not None and time.time() > exp

    # --- hash ops ---
    def hset(self, key: str, field: str, value: Any) -> None:
        self._hashes.setdefault(key, {})[field] = value

    def hget(self, key: str, field: str) -> Any | None:
        return self._hashes.get(key, {}).get(field)

    def hgetall(self, key: str) -> dict[str, Any]:
        return dict(self._hashes.get(key, {}))

    def hincrby(self, key: str, field: str, amount: int = 1) -> int:
        current = int(self._hashes.get(key, {}).get(field, 0))
        new = current + amount
        self.hset(key, field, new)
        return new

    # --- sorted set ops ---
    def zincrby(self, key: str, member: str, amount: float = 1.0) -> float:
        zset = self._zsets.setdefault(key, {})
        zset[member] = zset.get(member, 0.0) + amount
        return zset[member]

    def zrevrange(self, key: str, n: int) -> list[str]:
        zset = self._zsets.get(key, {})
        return [m for m, _ in sorted(zset.items(), key=lambda kv: kv[1], reverse=True)[:n]]

    def zsize(self, key: str) -> int:
        return len(self._zsets.get(key, {}))

    # --- test helper ---
    def reset(self) -> None:
        self._strings.clear()
        self._exp.clear()
        self._hashes.clear()
        self._zsets.clear()


store = InMemoryStore()
