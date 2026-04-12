"""Thin adapter layer for future /transcribe endpoint integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .pipeline import preprocess_for_scribe


def prepare_transcribe_audio(
    input_path: str | Path,
    working_dir: str | Path,
    *,
    job_id: str | None = None,
    timeout_s: float = 120,
) -> dict[str, Any]:
    """Return preprocessed WAV path and preprocessing telemetry for /transcribe."""

    result = preprocess_for_scribe(
        input_path=input_path,
        output_dir=working_dir,
        job_id=job_id,
        timeout_s=timeout_s,
    )

    return {
        "preprocessed_wav_path": str(result.output_path),
        "preprocessing_metrics": {
            "duration_s": result.duration_s,
            "input_sample_rate": result.input_sample_rate,
            "output_sample_rate": result.output_sample_rate,
            "channels": result.channels,
            "codec": result.codec,
            "ffmpeg_command": result.ffmpeg_command,
            "timings_ms": result.timings_ms,
        },
    }
