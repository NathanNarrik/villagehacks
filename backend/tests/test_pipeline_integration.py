from __future__ import annotations

import math
import shutil
import struct
import wave
from pathlib import Path

import pytest

from backend.audio_preprocess.errors import AudioProcessingTimeoutError, UnsupportedOrCorruptAudioError
from backend.audio_preprocess.pipeline import preprocess_for_scribe


FFMPEG_READY = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
pytestmark = pytest.mark.skipif(
    not FFMPEG_READY,
    reason="Integration tests require ffmpeg and ffprobe in PATH.",
)


def _write_tone(path: Path, sample_rate: int, duration_s: float = 1.0, hz: float = 440.0) -> None:
    total_samples = int(sample_rate * duration_s)

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)

        frames = bytearray()
        for idx in range(total_samples):
            sample = int(0.35 * 32767 * math.sin(2 * math.pi * hz * idx / sample_rate))
            frames.extend(struct.pack("<h", sample))

        wav.writeframes(frames)


def test_preprocess_8khz_telephony_audio_to_16khz_mono_pcm_wav(tmp_path: Path) -> None:
    input_path = tmp_path / "telephony_8k.wav"
    output_dir = tmp_path / "processed"
    _write_tone(input_path, sample_rate=8000)

    result = preprocess_for_scribe(input_path, output_dir, job_id="test_8k")

    assert result.output_path.exists()
    assert result.input_sample_rate == 8000
    assert result.output_sample_rate == 16000
    assert result.channels == 1
    assert result.codec == "pcm_s16le"

    with wave.open(str(result.output_path), "rb") as wav:
        assert wav.getframerate() == 16000
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2


def test_preprocess_non_8khz_input_also_normalizes_to_16khz(tmp_path: Path) -> None:
    input_path = tmp_path / "wideband_44k.wav"
    output_dir = tmp_path / "processed"
    _write_tone(input_path, sample_rate=44100)

    result = preprocess_for_scribe(input_path, output_dir, job_id="test_44k")

    assert result.output_path.exists()
    assert result.input_sample_rate == 44100
    assert result.output_sample_rate == 16000
    assert result.channels == 1
    assert result.codec == "pcm_s16le"


def test_corrupt_audio_raises_typed_error(tmp_path: Path) -> None:
    input_path = tmp_path / "corrupt.wav"
    output_dir = tmp_path / "processed"
    input_path.write_bytes(b"this is not valid audio")

    with pytest.raises(UnsupportedOrCorruptAudioError):
        preprocess_for_scribe(input_path, output_dir, job_id="test_corrupt")


def test_timeout_raises_typed_error(tmp_path: Path) -> None:
    input_path = tmp_path / "long_44k.wav"
    output_dir = tmp_path / "processed"
    _write_tone(input_path, sample_rate=44100, duration_s=12.0)

    with pytest.raises(AudioProcessingTimeoutError):
        preprocess_for_scribe(input_path, output_dir, job_id="test_timeout", timeout_s=1e-6)
