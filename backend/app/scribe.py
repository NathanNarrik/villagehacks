"""Layer 2 — ElevenLabs Scribe v2 batch transcription.

OWNED BY PERSON A. See HANDOFF_PERSON_A.md for the full spec.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx

from .config import settings
from .schemas import ScribeResult, ScribeWord

SCRIBE_BATCH_URL = "https://api.elevenlabs.io/v1/speech-to-text"


def _to_scribe_words(payload: dict[str, Any]) -> list[ScribeWord]:
    words_raw = payload.get("words")
    if not isinstance(words_raw, list):
        transcripts = payload.get("transcripts")
        if isinstance(transcripts, list) and transcripts:
            first = transcripts[0]
            if isinstance(first, dict):
                words_raw = first.get("words")
    if not isinstance(words_raw, list):
        return []

    out: list[ScribeWord] = []
    for item in words_raw:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        tok_type = str(item.get("type", "word")).lower()
        if tok_type != "word" or not text:
            continue

        try:
            start_ms = int(float(item.get("start", 0.0)) * 1000)
            end_ms = int(float(item.get("end", 0.0)) * 1000)
        except (TypeError, ValueError):
            continue
        if end_ms < start_ms:
            end_ms = start_ms

        out.append(
            ScribeWord(
                text=text,
                start_ms=start_ms,
                end_ms=end_ms,
                speaker_id=str(item.get("speaker_id") or "speaker_0"),
            )
        )

    return out


def _collect_audio_events(payload: dict[str, Any]) -> list[str]:
    words_blob = payload.get("words")
    if not isinstance(words_blob, list):
        transcripts = payload.get("transcripts")
        if isinstance(transcripts, list) and transcripts:
            first = transcripts[0]
            if isinstance(first, dict):
                words_blob = first.get("words")

    events: list[str] = []
    for item in words_blob if isinstance(words_blob, list) else []:
        if not isinstance(item, dict):
            continue
        tok_type = str(item.get("type", "")).lower()
        text = str(item.get("text", "")).strip()
        if tok_type in {"audio_event", "event"} and text:
            events.append(f"[{text.strip('()[]')}]")
    return events


async def transcribe_batch(wav_path: str, keyterms: list[str]) -> ScribeResult:
    """Submit cleaned audio to ElevenLabs Scribe v2 batch.

    Required call parameters:
        model_id="scribe_v2"
        diarize=True
        tag_audio_events=True
        timestamps_granularity="word"
        keywords=keyterms[:100]   # SDK caps at 100 per request

    Return a ScribeResult populated with:
        words[i].text         — the transcribed token
        words[i].start_ms     — int
        words[i].end_ms       — int
        words[i].speaker_id   — "speaker_0" / "speaker_1" (raw from ElevenLabs)
        audio_events          — flat list of [event] strings ("[coughing]", etc)
        duration_ms           — total clip length

    The pipeline maps speaker_0/speaker_1 to Doctor/Patient using a density heuristic
    in app/pipeline.py — Person A doesn't need to do that mapping here.
    """
    api_key = settings.elevenlabs_api_key()
    if not api_key:
        raise RuntimeError("ELEVENLABS_API_KEY is required for Scribe v2 batch transcription")

    path = Path(wav_path)
    if not path.exists():
        raise RuntimeError(f"Audio path not found for scribe transcription: {wav_path}")

    # ElevenLabs supports up to 1000 keyterms, but we keep this aligned with the
    # original handoff constraint for predictable request sizing in hackathon infra.
    limited_keyterms = [k.strip() for k in keyterms[:100] if k and k.strip()]

    data: list[tuple[str, str]] = [
        ("model_id", "scribe_v2"),
        ("diarize", "true"),
        ("tag_audio_events", "true"),
        ("timestamps_granularity", "word"),
        ("file_format", "pcm_s16le_16"),
    ]
    data.extend([("keyterms", term) for term in limited_keyterms])

    try:
        with path.open("rb") as f:
            files = {"file": (path.name, f, "audio/wav")}
            async with httpx.AsyncClient(timeout=90.0) as client:
                resp = await client.post(
                    SCRIBE_BATCH_URL,
                    headers={"xi-api-key": api_key},
                    params={"enable_logging": "true"},
                    data=data,
                    files=files,
                )
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:500] if exc.response is not None else str(exc)
        raise RuntimeError(f"Scribe batch failed ({exc.response.status_code}): {detail}") from exc
    except httpx.HTTPError as exc:
        raise RuntimeError(f"Scribe batch network failure: {exc}") from exc

    payload = resp.json()
    words = _to_scribe_words(payload)
    duration_ms = max((w.end_ms for w in words), default=0)
    audio_events = _collect_audio_events(payload)

    return ScribeResult(words=words, audio_events=audio_events, duration_ms=duration_ms)
