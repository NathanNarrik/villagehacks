"""Audio transcoding and verification helpers."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .errors import AudioToolingError
from .models import AudioExpectation, AudioMetadata


def resolve_binary(binary_name: str) -> str:
    """Resolve binary from PATH or raise a typed error."""

    resolved = shutil.which(binary_name)
    if not resolved:
        raise AudioToolingError(
            f"Required binary '{binary_name}' was not found in PATH. Install ffmpeg/ffprobe first."
        )
    return resolved


def transcode_to_pcm_wav(
    *,
    input_path: Path,
    output_path: Path,
    sample_rate: int,
    ffmpeg_bin: str,
    timeout_s: float,
    background_noise_profile: str | None = None,
) -> list[str]:
    """Transcode arbitrary input audio to mono pcm_s16le WAV at the target sample rate."""
    normalized_profile = (background_noise_profile or "").strip().lower()
    if normalized_profile:
        command = _build_noisy_transcode_command(
            input_path=input_path,
            output_path=output_path,
            sample_rate=sample_rate,
            ffmpeg_bin=ffmpeg_bin,
            background_noise_profile=normalized_profile,
        )
    else:
        command = [
            ffmpeg_bin,
            "-y",
            "-i",
            str(input_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ]

    try:
        proc = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise AudioToolingError(f"ffmpeg timed out after {timeout_s} seconds") from exc

    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        raise AudioToolingError(
            f"ffmpeg failed with code {proc.returncode} while writing '{output_path}': {stderr}"
        )

    return command


def transcode_with_rich_background(
    *,
    foreground_input_path: Path,
    conversation_input_paths: list[Path],
    output_path: Path,
    sample_rate: int,
    ffmpeg_bin: str,
    timeout_s: float,
    noise_profile: str,
) -> list[str]:
    """Transcode audio and apply richer background layers (babble + ambience + music)."""

    if not conversation_input_paths:
        raise AudioToolingError("transcode_with_rich_background requires conversation tracks")

    params = _rich_noise_params(noise_profile)
    command = _build_rich_noise_command(
        foreground_input_path=foreground_input_path,
        conversation_input_paths=conversation_input_paths,
        output_path=output_path,
        sample_rate=sample_rate,
        ffmpeg_bin=ffmpeg_bin,
        params=params,
    )

    try:
        proc = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise AudioToolingError(f"ffmpeg timed out after {timeout_s} seconds") from exc

    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        raise AudioToolingError(
            f"ffmpeg rich-noise mix failed with code {proc.returncode} for '{output_path}': {stderr}"
        )

    return command


def _build_noisy_transcode_command(
    *,
    input_path: Path,
    output_path: Path,
    sample_rate: int,
    ffmpeg_bin: str,
    background_noise_profile: str,
) -> list[str]:
    """Build ffmpeg command that adds synthetic background noise while transcoding."""

    amplitude = _noise_amplitude_for_profile(background_noise_profile)
    noise_source = (
        f"anoisesrc=color=pink:amplitude={amplitude:.3f}:sample_rate={sample_rate}"
    )
    filter_complex = (
        f"[0:a]aresample={sample_rate},aformat=sample_fmts=s16:channel_layouts=mono[voice];"
        "[1:a]aformat=sample_fmts=s16:channel_layouts=mono,highpass=f=90,lowpass=f=3600[noise];"
        "[voice][noise]amix=inputs=2:duration=first:dropout_transition=0,"
        "alimiter=limit=0.95[out]"
    )

    return [
        ffmpeg_bin,
        "-y",
        "-i",
        str(input_path),
        "-f",
        "lavfi",
        "-i",
        noise_source,
        "-filter_complex",
        filter_complex,
        "-map",
        "[out]",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]


def _build_rich_noise_command(
    *,
    foreground_input_path: Path,
    conversation_input_paths: list[Path],
    output_path: Path,
    sample_rate: int,
    ffmpeg_bin: str,
    params: dict[str, float],
) -> list[str]:
    command = [ffmpeg_bin, "-y", "-i", str(foreground_input_path)]
    for path in conversation_input_paths:
        command.extend(["-stream_loop", "-1", "-i", str(path)])

    conv_labels: list[str] = []
    conv_filters: list[str] = []
    tempo_options = [0.92, 1.04, 1.12]
    delay_options = [120, 280, 460]

    for idx, _ in enumerate(conversation_input_paths, start=1):
        label = f"conv{idx}"
        tempo = tempo_options[(idx - 1) % len(tempo_options)]
        delay = delay_options[(idx - 1) % len(delay_options)]
        conv_filters.append(
            (
                f"[{idx}:a]aresample={sample_rate},aformat=sample_fmts=fltp:channel_layouts=mono,"
                f"highpass=f=220,lowpass=f=3200,atempo={tempo:.2f},"
                f"volume={params['conversation_track_volume']:.3f},adelay={delay}|{delay}[{label}]"
            )
        )
        conv_labels.append(f"[{label}]")

    conv_mix = (
        "".join(conv_labels)
        + f"amix=inputs={len(conversation_input_paths)}:duration=longest:normalize=0,"
        f"volume={params['conversation_bus_volume']:.3f}[babble]"
    )

    filter_complex_parts = [
        (
            f"[0:a]aresample={sample_rate},aformat=sample_fmts=fltp:channel_layouts=mono,"
            "highpass=f=80,lowpass=f=7600[voice]"
        ),
        *conv_filters,
        conv_mix,
        (
            f"anoisesrc=color=pink:amplitude={params['ambient_amplitude']:.3f}:sample_rate={sample_rate},"
            "highpass=f=100,lowpass=f=5200[amb]"
        ),
        (
            f"sine=frequency=220:sample_rate={sample_rate},"
            f"volume={params['music_tone_1_volume']:.3f}[m1]"
        ),
        (
            f"sine=frequency=330:sample_rate={sample_rate},"
            f"volume={params['music_tone_2_volume']:.3f}[m2]"
        ),
        (
            "[m1][m2]amix=inputs=2:duration=longest:normalize=0,"
            "highpass=f=120,lowpass=f=1400[music]"
        ),
        (
            "[amb][music]amix=inputs=2:duration=longest:normalize=0,"
            f"volume={params['bed_volume']:.3f}[bed]"
        ),
        "[babble][bed]amix=inputs=2:duration=longest:normalize=0[noise]",
        (
            "[voice][noise]amix=inputs=2:duration=first:normalize=0:"
            f"weights='1 {params['final_noise_weight']:.3f}',alimiter=limit=0.95[out]"
        ),
    ]

    filter_complex = ";".join(filter_complex_parts)
    command.extend(
        [
            "-filter_complex",
            filter_complex,
            "-map",
            "[out]",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ]
    )
    return command


def _noise_amplitude_for_profile(noise_profile: str) -> float:
    """Return deterministic synthetic-noise amplitude for supported profiles."""

    if noise_profile == "high":
        return 0.110
    if noise_profile == "medium":
        return 0.065
    if noise_profile == "low":
        return 0.030
    return 0.050


def _rich_noise_params(noise_profile: str) -> dict[str, float]:
    normalized = (noise_profile or "").strip().lower()
    if normalized == "high":
        return {
            "conversation_track_volume": 0.22,
            "conversation_bus_volume": 1.00,
            "ambient_amplitude": 0.090,
            "music_tone_1_volume": 0.035,
            "music_tone_2_volume": 0.028,
            "bed_volume": 0.95,
            "final_noise_weight": 0.95,
        }
    if normalized == "medium":
        return {
            "conversation_track_volume": 0.16,
            "conversation_bus_volume": 0.92,
            "ambient_amplitude": 0.060,
            "music_tone_1_volume": 0.022,
            "music_tone_2_volume": 0.017,
            "bed_volume": 0.78,
            "final_noise_weight": 0.70,
        }
    return {
        "conversation_track_volume": 0.12,
        "conversation_bus_volume": 0.82,
        "ambient_amplitude": 0.040,
        "music_tone_1_volume": 0.016,
        "music_tone_2_volume": 0.012,
        "bed_volume": 0.62,
        "final_noise_weight": 0.52,
    }


def probe_audio(*, path: Path, ffprobe_bin: str, timeout_s: float) -> AudioMetadata:
    """Probe media metadata with ffprobe."""

    command = [
        ffprobe_bin,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(path),
    ]

    try:
        proc = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise AudioToolingError(f"ffprobe timed out after {timeout_s} seconds") from exc

    if proc.returncode != 0:
        raise AudioToolingError(
            f"ffprobe failed with code {proc.returncode} for '{path}': {proc.stderr.strip()}"
        )

    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise AudioToolingError(f"ffprobe returned invalid JSON for '{path}'") from exc

    return _parse_probe_payload(payload, path)


def _parse_probe_payload(payload: dict[str, Any], source_path: Path) -> AudioMetadata:
    streams = payload.get("streams", [])
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if not audio_stream:
        raise AudioToolingError(f"No audio stream detected in '{source_path}'.")

    fmt = payload.get("format", {})

    try:
        sample_rate = int(audio_stream["sample_rate"])
        channels = int(audio_stream["channels"])
        codec_name = str(audio_stream["codec_name"])
        format_name = str(fmt.get("format_name", ""))
        duration_raw = audio_stream.get("duration") or fmt.get("duration") or 0.0
        duration_s = float(duration_raw)
    except (KeyError, TypeError, ValueError) as exc:
        raise AudioToolingError(
            f"ffprobe metadata for '{source_path}' is incomplete or invalid."
        ) from exc

    return AudioMetadata(
        format_name=format_name,
        codec_name=codec_name,
        sample_rate=sample_rate,
        channels=channels,
        duration_s=duration_s,
    )


def verify_audio_file(
    *,
    path: Path,
    expected: AudioExpectation,
    ffprobe_bin: str,
    timeout_s: float,
) -> AudioMetadata:
    """Verify file exists and ffprobe metadata matches required format."""

    if not path.exists() or not path.is_file():
        raise AudioToolingError(f"Expected output file missing: '{path}'")

    metadata = probe_audio(path=path, ffprobe_bin=ffprobe_bin, timeout_s=timeout_s)

    if metadata.duration_s <= 0:
        raise AudioToolingError(f"Audio duration must be > 0 for '{path}'")

    container_names = {part.strip() for part in metadata.format_name.split(",") if part.strip()}
    if expected.container_name not in container_names:
        raise AudioToolingError(
            f"Expected container '{expected.container_name}' for '{path}', got '{metadata.format_name}'"
        )

    if metadata.codec_name != expected.codec_name:
        raise AudioToolingError(
            f"Expected codec '{expected.codec_name}' for '{path}', got '{metadata.codec_name}'"
        )

    if metadata.sample_rate != expected.sample_rate:
        raise AudioToolingError(
            f"Expected sample rate {expected.sample_rate} for '{path}', got {metadata.sample_rate}"
        )

    if metadata.channels != expected.channels:
        raise AudioToolingError(
            f"Expected {expected.channels} channel(s) for '{path}', got {metadata.channels}"
        )

    return metadata
