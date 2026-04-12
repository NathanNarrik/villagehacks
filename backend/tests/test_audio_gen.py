from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from backend.audio_gen.constants import MANIFEST_VERSION
from backend.audio_gen.errors import InputValidationError, ResumeGuardError
from backend.audio_gen.generator import (
    ClipProcessResult,
    GenerationConfig,
    RuntimeContext,
    _background_noise_profile_for_row,
    process_clip,
    run_generation,
)
from backend.audio_gen.io_utils import compute_input_hash, validate_resume_guard
from backend.audio_gen.models import AudioMetadata


class _DummyClient:
    def synthesize(self, *, text: str, voice_id: str, model_id: str) -> bytes:
        return b"dummy"


def _valid_row(clip_id: str) -> dict[str, str]:
    return {
        "clip_id": clip_id,
        "script_family_id": f"family_{clip_id}",
        "base_script_id": f"script_{clip_id}",
        "text": "example text",
        "voice_id": "voice_123",
        "voice_type": "neutral",
        "speech_style": "neutral",
        "accent": "us",
        "category": "general",
        "difficulty": "medium",
        "split": "train",
        "noise_level": "low",
        "has_interruptions": "false",
        "contains_numeric_confusion": "false",
        "numeric_confusion_type": "none",
        "contains_medical_terms": "false",
        "contains_ambiguity": "false",
        "scenario": "clean_speech",
        "scenario_group": "baseline",
        "noise_profile": "clean",
        "accent_profile": "",
        "medical_domain": "false",
        "medical_subtype": "",
    }


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_preflight_hard_fails_on_missing_columns(tmp_path: Path) -> None:
    input_path = tmp_path / "clips.csv"
    rows = [_valid_row("clip_1")]
    rows[0].pop("scenario")
    _write_csv(input_path, rows)

    config = GenerationConfig(
        input_path=input_path,
        out_dir=tmp_path / "out",
        concurrency=1,
        model_id="eleven_multilingual_v2",
        resume=False,
    )

    with pytest.raises(InputValidationError, match="missing required columns"):
        run_generation(config, _DummyClient())


def test_preflight_fails_on_invalid_voice_type_enum(tmp_path: Path) -> None:
    input_path = tmp_path / "clips.csv"
    rows = [_valid_row("clip_1")]
    rows[0]["voice_type"] = "telephony_noisy"
    _write_csv(input_path, rows)

    config = GenerationConfig(
        input_path=input_path,
        out_dir=tmp_path / "out",
        concurrency=1,
        model_id="eleven_multilingual_v2",
        resume=False,
    )

    with pytest.raises(InputValidationError, match="voice_type"):
        run_generation(config, _DummyClient())


def test_preflight_fails_on_medical_domain_category_mismatch(tmp_path: Path) -> None:
    input_path = tmp_path / "clips.csv"
    rows = [_valid_row("clip_1")]
    rows[0]["medical_domain"] = "true"
    rows[0]["category"] = "general"
    _write_csv(input_path, rows)

    config = GenerationConfig(
        input_path=input_path,
        out_dir=tmp_path / "out",
        concurrency=1,
        model_id="eleven_multilingual_v2",
        resume=False,
    )

    with pytest.raises(InputValidationError, match="medical_domain=true"):
        run_generation(config, _DummyClient())


def test_resume_guard_hash_mismatch_is_blocked(tmp_path: Path) -> None:
    input_path = tmp_path / "clips.csv"
    _write_csv(input_path, [_valid_row("clip_1")])

    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "manifest_version": MANIFEST_VERSION,
                "input_hash": "different_hash",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ResumeGuardError, match="input_hash mismatch"):
        validate_resume_guard(
            out_dir=out_dir,
            input_hash=compute_input_hash(input_path),
            resume=True,
        )


def test_resume_guard_hash_match_passes(tmp_path: Path) -> None:
    input_path = tmp_path / "clips.csv"
    _write_csv(input_path, [_valid_row("clip_1")])

    hash_value = compute_input_hash(input_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "manifest_version": MANIFEST_VERSION,
                "input_hash": hash_value,
            }
        ),
        encoding="utf-8",
    )

    validate_resume_guard(out_dir=out_dir, input_hash=hash_value, resume=True)


def test_run_continues_when_one_clip_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    input_path = tmp_path / "clips.csv"
    _write_csv(input_path, [_valid_row("clip_ok"), _valid_row("clip_fail")])

    def _fake_resolve_binary(_: str) -> str:
        return "binary"

    def _fake_process_clip(*args, **kwargs):  # type: ignore[no-untyped-def]
        row = args[0]
        if row["clip_id"] == "clip_fail":
            return ClipProcessResult(
                success=False,
                clip_row=None,
                error_row={
                    "clip_id": "clip_fail",
                    "manifest_version": MANIFEST_VERSION,
                    "input_hash": "placeholder",
                    "stage": "tts",
                    "error_class": "RuntimeError",
                    "error_message": "simulated",
                    "timestamp": "2026-01-01T00:00:00+00:00",
                },
            )

        return ClipProcessResult(
            success=True,
            clip_row={
                **row,
                "audio_clean_path": "clean/clip_ok.wav",
                "audio_telephony_path": "telephony/clip_ok.wav",
                "clean_sample_rate": 16000,
                "telephony_sample_rate": 8000,
                "clean_encoding": "pcm_s16le",
                "telephony_encoding": "pcm_s16le",
                "duration_sec": 1.0,
                "manifest_version": MANIFEST_VERSION,
                "input_hash": "placeholder",
            },
            error_row=None,
        )

    monkeypatch.setattr("backend.audio_gen.generator.resolve_binary", _fake_resolve_binary)
    monkeypatch.setattr("backend.audio_gen.generator.process_clip", _fake_process_clip)

    config = GenerationConfig(
        input_path=input_path,
        out_dir=tmp_path / "out",
        concurrency=2,
        model_id="eleven_multilingual_v2",
        resume=False,
    )

    summary = run_generation(config, _DummyClient())

    assert summary["successful_clips"] == 1
    assert summary["failed_clips"] == 1

    clips_lines = (tmp_path / "out" / "clips.jsonl").read_text(encoding="utf-8").strip().splitlines()
    error_lines = (tmp_path / "out" / "generation_errors.jsonl").read_text(
        encoding="utf-8"
    ).strip().splitlines()

    assert len(clips_lines) == 1
    assert len(error_lines) == 1


def test_process_clip_populates_split_audio_fields(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    row = _valid_row("clip_1")

    def _fake_synthesize_with_retry(**_: str) -> bytes:
        return b"fake-bytes"

    def _fake_transcode_to_pcm_wav(*, output_path: Path, **kwargs):  # type: ignore[no-untyped-def]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"audio")
        return ["ffmpeg"]

    def _fake_verify_audio_file(*, path: Path, **kwargs):  # type: ignore[no-untyped-def]
        if "clean" in path.parts:
            return AudioMetadata(
                format_name="wav",
                codec_name="pcm_s16le",
                sample_rate=16000,
                channels=1,
                duration_s=2.4,
            )
        return AudioMetadata(
            format_name="wav",
            codec_name="pcm_s16le",
            sample_rate=8000,
            channels=1,
            duration_s=2.4,
        )

    monkeypatch.setattr(
        "backend.audio_gen.generator._synthesize_with_retry",
        _fake_synthesize_with_retry,
    )
    monkeypatch.setattr(
        "backend.audio_gen.generator.transcode_to_pcm_wav",
        _fake_transcode_to_pcm_wav,
    )
    monkeypatch.setattr(
        "backend.audio_gen.generator.verify_audio_file",
        _fake_verify_audio_file,
    )

    runtime = RuntimeContext(
        client=_DummyClient(),
        model_id="eleven_multilingual_v2",
        input_hash="hash123",
        out_dir=tmp_path,
        ffmpeg_bin="ffmpeg",
        ffprobe_bin="ffprobe",
        timeout_s=5,
    )

    result = process_clip(
        row,
        runtime,
        raw_dir=tmp_path / "raw",
        clean_dir=tmp_path / "clean",
        telephony_dir=tmp_path / "telephony",
    )

    assert result.success
    assert result.clip_row is not None
    assert result.clip_row["audio_clean_path"] == "clean/clip_1.wav"
    assert result.clip_row["audio_telephony_path"] == "telephony/clip_1.wav"
    assert result.clip_row["clean_sample_rate"] == 16000
    assert result.clip_row["telephony_sample_rate"] == 8000
    assert result.clip_row["clean_encoding"] == "pcm_s16le"
    assert result.clip_row["telephony_encoding"] == "pcm_s16le"


def test_process_clip_applies_stronger_noise_to_noisy_scenarios(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    row = _valid_row("clip_noisy")
    row["scenario"] = "noisy_environment"
    row["noise_profile"] = "high"
    row["noise_level"] = "high"

    captured_noise_profiles: list[str | None] = []

    def _fake_synthesize_with_retry(**_: str) -> bytes:
        return b"fake-bytes"

    def _fake_transcode_to_pcm_wav(*, output_path: Path, **kwargs):  # type: ignore[no-untyped-def]
        captured_noise_profiles.append(kwargs.get("background_noise_profile"))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"audio")
        return ["ffmpeg"]

    def _fake_verify_audio_file(*, path: Path, **kwargs):  # type: ignore[no-untyped-def]
        if "clean" in path.parts:
            return AudioMetadata(
                format_name="wav",
                codec_name="pcm_s16le",
                sample_rate=16000,
                channels=1,
                duration_s=2.0,
            )
        return AudioMetadata(
            format_name="wav",
            codec_name="pcm_s16le",
            sample_rate=8000,
            channels=1,
            duration_s=2.0,
        )

    monkeypatch.setattr(
        "backend.audio_gen.generator._synthesize_with_retry",
        _fake_synthesize_with_retry,
    )
    monkeypatch.setattr(
        "backend.audio_gen.generator.transcode_to_pcm_wav",
        _fake_transcode_to_pcm_wav,
    )
    monkeypatch.setattr(
        "backend.audio_gen.generator.verify_audio_file",
        _fake_verify_audio_file,
    )

    runtime = RuntimeContext(
        client=_DummyClient(),
        model_id="eleven_multilingual_v2",
        input_hash="hash123",
        out_dir=tmp_path,
        ffmpeg_bin="ffmpeg",
        ffprobe_bin="ffprobe",
        timeout_s=5,
    )

    result = process_clip(
        row,
        runtime,
        raw_dir=tmp_path / "raw",
        clean_dir=tmp_path / "clean",
        telephony_dir=tmp_path / "telephony",
    )

    assert result.success
    assert captured_noise_profiles[0] == "high"
    assert captured_noise_profiles[1] is None


def test_process_clip_logs_verify_failures(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    row = _valid_row("clip_bad_verify")

    def _fake_synthesize_with_retry(**_: str) -> bytes:
        return b"fake-bytes"

    def _fake_transcode_to_pcm_wav(*, output_path: Path, **kwargs):  # type: ignore[no-untyped-def]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"audio")
        return ["ffmpeg"]

    def _fake_verify_audio_file(**kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("duration is zero")

    monkeypatch.setattr(
        "backend.audio_gen.generator._synthesize_with_retry",
        _fake_synthesize_with_retry,
    )
    monkeypatch.setattr(
        "backend.audio_gen.generator.transcode_to_pcm_wav",
        _fake_transcode_to_pcm_wav,
    )
    monkeypatch.setattr(
        "backend.audio_gen.generator.verify_audio_file",
        _fake_verify_audio_file,
    )

    runtime = RuntimeContext(
        client=_DummyClient(),
        model_id="eleven_multilingual_v2",
        input_hash="hash123",
        out_dir=tmp_path,
        ffmpeg_bin="ffmpeg",
        ffprobe_bin="ffprobe",
        timeout_s=5,
    )

    result = process_clip(
        row,
        runtime,
        raw_dir=tmp_path / "raw",
        clean_dir=tmp_path / "clean",
        telephony_dir=tmp_path / "telephony",
    )

    assert not result.success
    assert result.error_row is not None
    assert result.error_row["stage"] == "clean_verify"


def test_template_files_include_schema_version(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "clips.csv"
    _write_csv(input_path, [_valid_row("clip_1")])

    def _fake_resolve_binary(_: str) -> str:
        return "binary"

    def _fake_process_clip(*args, **kwargs):  # type: ignore[no-untyped-def]
        row = args[0]
        return ClipProcessResult(
            success=True,
            clip_row={
                **row,
                "audio_clean_path": "clean/clip_1.wav",
                "audio_telephony_path": "telephony/clip_1.wav",
                "clean_sample_rate": 16000,
                "telephony_sample_rate": 8000,
                "clean_encoding": "pcm_s16le",
                "telephony_encoding": "pcm_s16le",
                "duration_sec": 1.0,
                "manifest_version": MANIFEST_VERSION,
                "input_hash": "placeholder",
            },
            error_row=None,
        )

    monkeypatch.setattr("backend.audio_gen.generator.resolve_binary", _fake_resolve_binary)
    monkeypatch.setattr("backend.audio_gen.generator.process_clip", _fake_process_clip)

    config = GenerationConfig(
        input_path=input_path,
        out_dir=tmp_path / "out",
        concurrency=1,
        model_id="eleven_multilingual_v2",
        resume=False,
    )

    run_generation(config, _DummyClient())

    for name in (
        "word_features.template.jsonl",
        "numeric_features.template.jsonl",
        "medical_entities.template.jsonl",
    ):
        payload = json.loads((tmp_path / "out" / name).read_text(encoding="utf-8").strip())
        assert payload["schema_version"] == MANIFEST_VERSION


def test_background_noise_profile_mapping() -> None:
    base = _valid_row("clip_mapping")
    assert _background_noise_profile_for_row(base) is None

    noisy = dict(base)
    noisy["scenario"] = "noisy_environment"
    noisy["noise_profile"] = "medium"
    assert _background_noise_profile_for_row(noisy) == "medium"

    medical = dict(base)
    medical["scenario"] = "medical_conversation"
    medical["noise_profile"] = "clean"
    medical["noise_level"] = "high"
    assert _background_noise_profile_for_row(medical) == "high"
