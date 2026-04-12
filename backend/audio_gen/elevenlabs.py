"""Minimal ElevenLabs API client with typed errors."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib import error as urlerror
from urllib import parse, request

from .errors import ElevenLabsAPIError


@dataclass(frozen=True)
class ElevenLabsClient:
    """HTTP client for ElevenLabs text-to-speech endpoint."""

    api_key: str
    base_url: str = "https://api.elevenlabs.io"
    timeout_s: float = 60.0

    def synthesize(self, *, text: str, voice_id: str, model_id: str) -> bytes:
        """Synthesize speech for text and return audio bytes."""

        if not text.strip():
            raise ElevenLabsAPIError("Cannot synthesize empty text.")

        encoded_voice = parse.quote(voice_id, safe="")
        url = f"{self.base_url}/v1/text-to-speech/{encoded_voice}?output_format=mp3_44100_128"

        payload: dict[str, Any] = {
            "text": text,
            "model_id": model_id,
        }

        req = request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "xi-api-key": self.api_key,
                "Content-Type": "application/json",
                "Accept": "audio/mpeg",
            },
        )

        try:
            with request.urlopen(req, timeout=self.timeout_s) as resp:
                body = resp.read()
                if not body:
                    raise ElevenLabsAPIError(
                        "ElevenLabs returned an empty audio payload.",
                        status_code=getattr(resp, "status", None),
                    )
                return body
        except urlerror.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise ElevenLabsAPIError(
                f"ElevenLabs API failed ({exc.code}): {details}",
                status_code=exc.code,
            ) from exc
        except urlerror.URLError as exc:
            raise ElevenLabsAPIError(f"Network error calling ElevenLabs: {exc.reason}") from exc
