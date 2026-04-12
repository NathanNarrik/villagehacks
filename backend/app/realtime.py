"""Helpers for realtime Scribe v2 relay and stream-token management."""
from __future__ import annotations

import base64
import json
import logging
import secrets
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

from .config import settings
from .schemas import ScribeWord
from .storage import store

log = logging.getLogger(__name__)

STREAM_TOKEN_KEY_PREFIX = "cc:stream_token:"


def issue_stream_token(ttl_sec: int | None = None) -> dict[str, Any]:
    """Create a single-use stream token stored in the backend cache."""
    token = secrets.token_urlsafe(32)
    expires = int(ttl_sec if ttl_sec is not None else settings.STREAM_TOKEN_TTL_SEC)
    store.set(f"{STREAM_TOKEN_KEY_PREFIX}{token}", {"valid": True}, ttl_sec=expires)
    return {"token": token, "expires_in": expires}


def consume_stream_token(token: str) -> bool:
    if not token:
        return False
    key = f"{STREAM_TOKEN_KEY_PREFIX}{token}"
    found = store.get(key)
    if not found:
        return False
    # Single-use: remove immediately after first successful check.
    store.delete(key)
    return True


def realtime_dependency_status() -> str:
    if not settings.elevenlabs_api_key():
        return "not_configured"
    try:
        import websockets  # noqa: F401
    except Exception:
        return "missing_websockets_dependency"
    return "ready"


def _secs_to_ms(value: Any) -> int:
    try:
        return max(0, int(float(value) * 1000))
    except (TypeError, ValueError):
        return 0


def frontend_words_from_realtime_event(event: dict[str, Any]) -> list[dict[str, Any]]:
    words = event.get("words")
    if not isinstance(words, list):
        return []
    out: list[dict[str, Any]] = []
    for raw in words:
        if not isinstance(raw, dict):
            continue
        if str(raw.get("type", "word")).lower() != "word":
            continue
        text = str(raw.get("text", "")).strip()
        if not text:
            continue
        out.append(
            {
                "word": text,
                "start_ms": _secs_to_ms(raw.get("start")),
                "end_ms": _secs_to_ms(raw.get("end")),
            }
        )
    return out


def scribe_words_from_realtime_event(event: dict[str, Any]) -> list[ScribeWord]:
    """Convert committed realtime event payload into internal ScribeWord objects."""
    frontend_words = frontend_words_from_realtime_event(event)
    if frontend_words:
        return [
            ScribeWord(
                text=str(w["word"]),
                start_ms=int(w["start_ms"]),
                end_ms=int(w["end_ms"]),
                speaker_id="speaker_0",
            )
            for w in frontend_words
        ]

    # Fallback when timestamps are unavailable: keep flow alive with synthetic timing.
    text = str(event.get("text", "")).strip()
    if not text:
        return []
    synthetic: list[ScribeWord] = []
    cursor = 0
    for tok in text.split():
        start = cursor
        duration = max(80, len(tok) * 40)
        end = start + duration
        synthetic.append(
            ScribeWord(text=tok, start_ms=start, end_ms=end, speaker_id="speaker_0")
        )
        cursor = end + 20
    return synthetic


def realtime_error_payload(event: dict[str, Any]) -> dict[str, str]:
    return {
        "type": "error",
        "stage": "scribe_realtime",
        "message": str(event.get("message") or event.get("error") or "Unknown realtime error"),
    }


@dataclass
class ScribeRealtimeClient:
    """Small websocket client for ElevenLabs Scribe realtime relay."""

    api_key: str
    model_id: str = settings.SCRIBE_REALTIME_MODEL_ID
    _ws: Any | None = None

    async def connect(self) -> None:
        try:
            import websockets  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("websockets package not available") from exc

        query = urlencode(
            {
                "model_id": self.model_id,
                "audio_format": "pcm_16000",
                "include_timestamps": "true",
                "commit_strategy": "vad",
                "vad_silence_threshold_secs": "0.8",
            }
        )
        ws_url = f"{settings.SCRIBE_REALTIME_WS_URL}?{query}"
        headers = {"xi-api-key": self.api_key}

        try:
            self._ws = await websockets.connect(  # type: ignore[attr-defined]
                ws_url,
                additional_headers=headers,
                max_size=2**22,
            )
        except TypeError:
            # Older websockets versions use extra_headers instead.
            self._ws = await websockets.connect(  # type: ignore[attr-defined]
                ws_url,
                extra_headers=headers,
                max_size=2**22,
            )

    async def send_audio_chunk(self, chunk: bytes, *, commit: bool = False) -> None:
        if self._ws is None:
            raise RuntimeError("Realtime upstream websocket is not connected")
        payload = {
            "message_type": "input_audio_chunk",
            "audio_base_64": base64.b64encode(chunk).decode("ascii"),
            "sample_rate": 16000,
            "commit": commit,
        }
        await self._ws.send(json.dumps(payload))

    async def commit(self) -> None:
        await self.send_audio_chunk(b"", commit=True)

    async def recv(self) -> dict[str, Any]:
        if self._ws is None:
            raise RuntimeError("Realtime upstream websocket is not connected")
        message = await self._ws.recv()
        if isinstance(message, bytes):
            message = message.decode("utf-8", errors="replace")
        return json.loads(message)

    async def close(self) -> None:
        if self._ws is not None:
            await self._ws.close()
            self._ws = None
