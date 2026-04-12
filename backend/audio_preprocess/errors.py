"""Typed exceptions for the audio preprocessing pipeline."""

from __future__ import annotations


class AudioPreprocessError(Exception):
    """Base exception for all preprocessing failures."""


class FFmpegNotFoundError(AudioPreprocessError):
    """Raised when ffmpeg/ffprobe cannot be found in PATH."""


class AudioProbeError(AudioPreprocessError):
    """Raised when input/output media metadata cannot be probed."""


class UnsupportedOrCorruptAudioError(AudioPreprocessError):
    """Raised when audio is corrupt or unsupported by ffmpeg/ffprobe."""


class AudioFormatValidationError(AudioPreprocessError):
    """Raised when processed output does not match required Scribe format."""


class AudioProcessingTimeoutError(AudioPreprocessError):
    """Raised when ffprobe/ffmpeg exceeds the configured timeout."""

    def __init__(self, stage: str, timeout_s: float) -> None:
        super().__init__(f"{stage} exceeded timeout of {timeout_s} seconds")
        self.stage = stage
        self.timeout_s = timeout_s


class AudioProcessingFailedError(AudioPreprocessError):
    """Raised when ffmpeg exits with a non-zero status."""

    def __init__(
        self,
        message: str,
        *,
        returncode: int,
        command: list[str],
        stderr: str,
    ) -> None:
        super().__init__(message)
        self.returncode = returncode
        self.command = command
        self.stderr = stderr
