"""Dataclasses for audio generation metadata."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AudioMetadata:
    """Audio metadata returned by ffprobe."""

    format_name: str
    codec_name: str
    sample_rate: int
    channels: int
    duration_s: float


@dataclass(frozen=True)
class AudioExpectation:
    """Expected format metadata used for post-write verification."""

    sample_rate: int
    codec_name: str
    channels: int
    container_name: str
