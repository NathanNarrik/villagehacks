"""Typed errors for audio generation pipeline."""

from __future__ import annotations


class AudioGenError(Exception):
    """Base class for audio generation errors."""


class InputValidationError(AudioGenError):
    """Raised when source metadata fails preflight validation."""


class ResumeGuardError(AudioGenError):
    """Raised when resume metadata does not match the current input."""


class ElevenLabsAPIError(AudioGenError):
    """Raised when ElevenLabs API calls fail."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class AudioToolingError(AudioGenError):
    """Raised when ffmpeg/ffprobe processing fails."""


class ClipStageError(AudioGenError):
    """Raised when one clip fails at a specific stage."""

    def __init__(self, stage: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage
