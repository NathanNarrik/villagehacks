"""Layer 2 — ElevenLabs Scribe v2 batch transcription.

OWNED BY PERSON A. See HANDOFF_PERSON_A.md for the full spec.
"""
from __future__ import annotations

from .schemas import ScribeResult


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
    raise NotImplementedError("scribe.transcribe_batch — Person A: see HANDOFF_PERSON_A.md")
