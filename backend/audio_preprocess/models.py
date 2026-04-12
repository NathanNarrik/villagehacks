"""Typed models for audio preprocessing results and metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AudioMetadata:
    """Minimal audio metadata required by preprocessing validation."""

    format_name: str
    codec_name: str
    sample_rate: int
    channels: int
    duration_s: float


@dataclass(frozen=True)
class PreprocessResult:
    """Result contract for downstream transcription pipeline stages."""

    output_path: Path
    duration_s: float
    input_sample_rate: int
    output_sample_rate: int
    channels: int
    codec: str
    ffmpeg_command: list[str]
    timings_ms: dict[str, int]
