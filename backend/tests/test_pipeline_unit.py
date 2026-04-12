from __future__ import annotations

from pathlib import Path

import pytest

from backend.audio_preprocess.errors import AudioFormatValidationError
from backend.audio_preprocess.models import AudioMetadata
from backend.audio_preprocess.pipeline import FILTER_CHAIN, build_ffmpeg_command, validate_output_metadata


def test_build_ffmpeg_command_uses_exact_filter_order_and_output_flags() -> None:
    command = build_ffmpeg_command(
        Path("/tmp/input.wav"),
        Path("/tmp/output.wav"),
        ffmpeg_bin="ffmpeg",
    )

    assert command == [
        "ffmpeg",
        "-y",
        "-i",
        "/tmp/input.wav",
        "-af",
        FILTER_CHAIN,
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        "/tmp/output.wav",
    ]

    filter_order = command[command.index("-af") + 1].split(",")
    assert filter_order == [
        "loudnorm=I=-16:LRA=11:TP=-1.5",
        "afftdn=nf=-25",
        "aresample=16000:resampler=soxr",
    ]


def test_validate_output_metadata_rejects_non_mono_output() -> None:
    non_mono_output = AudioMetadata(
        format_name="wav",
        codec_name="pcm_s16le",
        sample_rate=16000,
        channels=2,
        duration_s=1.0,
    )

    with pytest.raises(AudioFormatValidationError, match="mono"):
        validate_output_metadata(non_mono_output)
