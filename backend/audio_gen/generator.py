"""Batch generation pipeline for ElevenLabs dataset audio."""

from __future__ import annotations

import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .audio import resolve_binary, transcode_to_pcm_wav, verify_audio_file
from .constants import (
    CLEAN_SAMPLE_RATE,
    MANIFEST_VERSION,
    MONO_CHANNELS,
    PCM_CODEC,
    TELEPHONY_SAMPLE_RATE,
    WAV_CONTAINER,
)
from .elevenlabs import ElevenLabsClient
from .errors import ClipStageError, ElevenLabsAPIError
from .io_utils import (
    compute_input_hash,
    load_input_rows,
    load_successful_clip_ids,
    output_files_for_run,
    validate_resume_guard,
    validate_rows,
    write_run_metadata,
    write_template_files,
)
from .models import AudioExpectation

MAX_RETRIES = 5


@dataclass(frozen=True)
class GenerationConfig:
    """Configuration for dataset generation."""

    input_path: Path
    out_dir: Path
    concurrency: int
    model_id: str
    resume: bool
    timeout_s: float = 120.0


@dataclass(frozen=True)
class ClipProcessResult:
    """Result payload for a single clip generation attempt."""

    success: bool
    clip_row: dict[str, Any] | None
    error_row: dict[str, Any] | None


@dataclass(frozen=True)
class RuntimeContext:
    """Runtime dependencies shared by worker threads."""

    client: ElevenLabsClient
    model_id: str
    input_hash: str
    out_dir: Path
    ffmpeg_bin: str
    ffprobe_bin: str
    timeout_s: float


def run_generation(config: GenerationConfig, client: ElevenLabsClient) -> dict[str, Any]:
    """Run batch generation and write dataset artifacts."""

    config.out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = config.out_dir / "raw"
    clean_dir = config.out_dir / "clean"
    telephony_dir = config.out_dir / "telephony"
    raw_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)
    telephony_dir.mkdir(parents=True, exist_ok=True)

    source_rows = load_input_rows(config.input_path)
    rows = validate_rows(source_rows)

    input_hash = compute_input_hash(config.input_path)
    validate_resume_guard(out_dir=config.out_dir, input_hash=input_hash, resume=config.resume)

    ffmpeg_bin = resolve_binary("ffmpeg")
    ffprobe_bin = resolve_binary("ffprobe")

    runtime = RuntimeContext(
        client=client,
        model_id=config.model_id,
        input_hash=input_hash,
        out_dir=config.out_dir,
        ffmpeg_bin=ffmpeg_bin,
        ffprobe_bin=ffprobe_bin,
        timeout_s=config.timeout_s,
    )

    write_template_files(out_dir=config.out_dir)

    existing_success = (
        load_successful_clip_ids(out_dir=config.out_dir, input_hash=input_hash)
        if config.resume
        else set()
    )

    clips_path, errors_path, write_mode = output_files_for_run(
        out_dir=config.out_dir,
        resume=config.resume,
    )

    submitted_rows = [row for row in rows if row["clip_id"] not in existing_success]
    skipped_count = len(rows) - len(submitted_rows)

    started_at = _now_iso()
    success_count = 0
    failed_count = 0

    with (
        clips_path.open(write_mode, encoding="utf-8") as clips_file,
        errors_path.open(write_mode, encoding="utf-8") as errors_file,
        ThreadPoolExecutor(max_workers=config.concurrency) as pool,
    ):
        futures = {
            pool.submit(
                process_clip,
                row,
                runtime,
                raw_dir,
                clean_dir,
                telephony_dir,
            ): row["clip_id"]
            for row in submitted_rows
        }

        for future in as_completed(futures):
            clip_id = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # pragma: no cover - defensive fallback
                failed_count += 1
                errors_file.write(
                    json.dumps(
                        _error_record(
                            clip_id=clip_id,
                            input_hash=input_hash,
                            stage="internal",
                            error_class=type(exc).__name__,
                            error_message=str(exc),
                        )
                    )
                    + "\n"
                )
                errors_file.flush()
                continue

            if result.success and result.clip_row:
                success_count += 1
                clips_file.write(json.dumps(result.clip_row) + "\n")
                clips_file.flush()
                continue

            failed_count += 1
            if result.error_row:
                errors_file.write(json.dumps(result.error_row) + "\n")
                errors_file.flush()

    completed_at = _now_iso()
    run_metadata = {
        "manifest_version": MANIFEST_VERSION,
        "input_hash": input_hash,
        "input_path": str(config.input_path.resolve()),
        "resume": config.resume,
        "started_at": started_at,
        "completed_at": completed_at,
        "total_clips": len(rows),
        "submitted_clips": len(submitted_rows),
        "skipped_clips": skipped_count,
        "successful_clips": success_count,
        "failed_clips": failed_count,
    }
    write_run_metadata(out_dir=config.out_dir, payload=run_metadata)
    return run_metadata


def process_clip(
    row: dict[str, Any],
    runtime: RuntimeContext,
    raw_dir: Path,
    clean_dir: Path,
    telephony_dir: Path,
) -> ClipProcessResult:
    """Process one clip end-to-end and return a success/error record."""

    clip_id = str(row["clip_id"])
    safe_name = _safe_filename(clip_id)

    raw_path = raw_dir / f"{safe_name}.source"
    clean_path = clean_dir / f"{safe_name}.wav"
    telephony_path = telephony_dir / f"{safe_name}.wav"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    telephony_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        audio_bytes = _synthesize_with_retry(
            client=runtime.client,
            text=str(row["text"]),
            voice_id=str(row["voice_id"]),
            model_id=runtime.model_id,
        )
    except Exception as exc:
        return ClipProcessResult(
            success=False,
            clip_row=None,
            error_row=_from_exception(
                clip_id=clip_id,
                input_hash=runtime.input_hash,
                stage="tts",
                exc=exc,
            ),
        )

    try:
        raw_path.write_bytes(audio_bytes)
    except Exception as exc:
        return ClipProcessResult(
            success=False,
            clip_row=None,
            error_row=_from_exception(
                clip_id=clip_id,
                input_hash=runtime.input_hash,
                stage="write_raw",
                exc=exc,
            ),
        )

    try:
        noisy_profile = _background_noise_profile_for_row(row)
        transcode_to_pcm_wav(
            input_path=raw_path,
            output_path=clean_path,
            sample_rate=CLEAN_SAMPLE_RATE,
            ffmpeg_bin=runtime.ffmpeg_bin,
            timeout_s=runtime.timeout_s,
            background_noise_profile=noisy_profile,
        )
    except Exception as exc:
        return ClipProcessResult(
            success=False,
            clip_row=None,
            error_row=_from_exception(
                clip_id=clip_id,
                input_hash=runtime.input_hash,
                stage="clean_transcode",
                exc=exc,
            ),
        )

    clean_expected = AudioExpectation(
        sample_rate=CLEAN_SAMPLE_RATE,
        codec_name=PCM_CODEC,
        channels=MONO_CHANNELS,
        container_name=WAV_CONTAINER,
    )

    try:
        clean_meta = verify_audio_file(
            path=clean_path,
            expected=clean_expected,
            ffprobe_bin=runtime.ffprobe_bin,
            timeout_s=runtime.timeout_s,
        )
    except Exception as exc:
        return ClipProcessResult(
            success=False,
            clip_row=None,
            error_row=_from_exception(
                clip_id=clip_id,
                input_hash=runtime.input_hash,
                stage="clean_verify",
                exc=exc,
            ),
        )

    try:
        transcode_to_pcm_wav(
            input_path=clean_path,
            output_path=telephony_path,
            sample_rate=TELEPHONY_SAMPLE_RATE,
            ffmpeg_bin=runtime.ffmpeg_bin,
            timeout_s=runtime.timeout_s,
        )
    except Exception as exc:
        return ClipProcessResult(
            success=False,
            clip_row=None,
            error_row=_from_exception(
                clip_id=clip_id,
                input_hash=runtime.input_hash,
                stage="telephony_transcode",
                exc=exc,
            ),
        )

    telephony_expected = AudioExpectation(
        sample_rate=TELEPHONY_SAMPLE_RATE,
        codec_name=PCM_CODEC,
        channels=MONO_CHANNELS,
        container_name=WAV_CONTAINER,
    )

    try:
        verify_audio_file(
            path=telephony_path,
            expected=telephony_expected,
            ffprobe_bin=runtime.ffprobe_bin,
            timeout_s=runtime.timeout_s,
        )
    except Exception as exc:
        return ClipProcessResult(
            success=False,
            clip_row=None,
            error_row=_from_exception(
                clip_id=clip_id,
                input_hash=runtime.input_hash,
                stage="telephony_verify",
                exc=exc,
            ),
        )

    clip_row = dict(row)
    clip_row.update(
        {
            "audio_clean_path": _relative_to_out(clean_path, runtime.out_dir),
            "audio_telephony_path": _relative_to_out(telephony_path, runtime.out_dir),
            "clean_sample_rate": CLEAN_SAMPLE_RATE,
            "telephony_sample_rate": TELEPHONY_SAMPLE_RATE,
            "clean_encoding": PCM_CODEC,
            "telephony_encoding": PCM_CODEC,
            "duration_sec": round(clean_meta.duration_s, 6),
            "manifest_version": MANIFEST_VERSION,
            "input_hash": runtime.input_hash,
        }
    )

    return ClipProcessResult(success=True, clip_row=clip_row, error_row=None)


def _synthesize_with_retry(
    *,
    client: ElevenLabsClient,
    text: str,
    voice_id: str,
    model_id: str,
) -> bytes:
    last_exc: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return client.synthesize(text=text, voice_id=voice_id, model_id=model_id)
        except ElevenLabsAPIError as exc:
            last_exc = exc
            if not _is_retryable(exc) or attempt == MAX_RETRIES:
                break
            _sleep_for_retry(attempt)
        except Exception as exc:
            last_exc = exc
            if attempt == MAX_RETRIES:
                break
            _sleep_for_retry(attempt)

    if last_exc is None:  # pragma: no cover - defensive
        raise ClipStageError("tts", "Unknown ElevenLabs synthesis failure")

    raise last_exc


def _is_retryable(exc: ElevenLabsAPIError) -> bool:
    if exc.status_code is None:
        return True
    return exc.status_code == 429 or exc.status_code >= 500


def _sleep_for_retry(attempt: int) -> None:
    base = 0.5 * (2 ** (attempt - 1))
    jitter = random.uniform(0.0, 0.25)
    time.sleep(base + jitter)


def _relative_to_out(path: Path, out_dir: Path) -> str:
    return path.resolve().relative_to(out_dir.resolve()).as_posix()


def _safe_filename(raw: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", raw).strip("_")
    return cleaned or "clip"


def _background_noise_profile_for_row(row: dict[str, Any]) -> str | None:
    """Map row-level labels to synthetic background-noise injection profile."""

    scenario = str(row.get("scenario", "")).strip().lower()
    noise_profile = str(row.get("noise_profile", "")).strip().lower()
    noise_level = str(row.get("noise_level", "")).strip().lower()

    if scenario not in {"noisy_environment", "medical_conversation"}:
        return None

    if noise_profile in {"medium", "high"}:
        return noise_profile

    if noise_level in {"medium", "high"}:
        return noise_level

    return None


def _from_exception(*, clip_id: str, input_hash: str, stage: str, exc: Exception) -> dict[str, Any]:
    if isinstance(exc, ClipStageError):
        stage = exc.stage

    return _error_record(
        clip_id=clip_id,
        input_hash=input_hash,
        stage=stage,
        error_class=type(exc).__name__,
        error_message=str(exc),
    )


def _error_record(
    *,
    clip_id: str,
    input_hash: str,
    stage: str,
    error_class: str,
    error_message: str,
) -> dict[str, Any]:
    return {
        "clip_id": clip_id,
        "manifest_version": MANIFEST_VERSION,
        "input_hash": input_hash,
        "stage": stage,
        "error_class": error_class,
        "error_message": error_message,
        "timestamp": _now_iso(),
    }


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()
