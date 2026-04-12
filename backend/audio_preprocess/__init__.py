"""Public interface for Scribe audio preprocessing."""

from .adapter import prepare_transcribe_audio
from .errors import (
    AudioFormatValidationError,
    AudioPreprocessError,
    AudioProbeError,
    AudioProcessingFailedError,
    AudioProcessingTimeoutError,
    FFmpegNotFoundError,
    UnsupportedOrCorruptAudioError,
)
from .models import PreprocessResult
from .pipeline import preprocess_for_scribe

__all__ = [
    "preprocess_for_scribe",
    "prepare_transcribe_audio",
    "PreprocessResult",
    "AudioPreprocessError",
    "FFmpegNotFoundError",
    "AudioProbeError",
    "UnsupportedOrCorruptAudioError",
    "AudioFormatValidationError",
    "AudioProcessingTimeoutError",
    "AudioProcessingFailedError",
]
