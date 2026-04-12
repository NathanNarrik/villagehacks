"""Generate and export the multi-variant demo audio catalog."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .env_utils import load_audio_gen_env, resolve_elevenlabs_api_key
from .elevenlabs import ElevenLabsClient
from .generator import GenerationConfig, run_generation
from .io_utils import load_input_rows, validate_rows
from .remix_rich_noise import remix_run_dir

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_INPUT_PATH = REPO_ROOT / "backend" / "audio_gen" / "input" / "demo_cards_20260412.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "backend" / "audio_gen" / "output" / "demo_cards_20260412"
DEFAULT_MANIFEST_PATH = REPO_ROOT / "backend" / "test_audio" / "demo" / "manifest.csv"
DEFAULT_SCRIPTS_DIR = REPO_ROOT / "backend" / "test_audio" / "demo" / "scripts"
DEFAULT_BACKEND_AUDIO_DIR = REPO_ROOT / "backend" / "test_audio" / "demo" / "audio"
DEFAULT_FRONTEND_PUBLIC_DIR = REPO_ROOT / "frontend" / "public"

MANIFEST_FIELDNAMES = [
    "demo_id",
    "situation_id",
    "situation",
    "variant_id",
    "variant_label",
    "audio_relpath",
    "frontend_public_relpath",
    "script_ref",
    "expected_highlights",
    "notes",
]

VOICE_ROTATIONS = {
    "clean": [
        "CwhRBWXzGAHq8TQ4Fs17",
        "pNInz6obpgDQGcFmaJgB",
        "JBFqnCBsd6RMkjVDRZzb",
        "EXAVITQu4vr4xnSDxMaL",
        "N2lVS1w4EtoT3dr4eOWO",
        "onwK4e9ZLuTAKqWW03F9",
    ],
    "ambient": [
        "EXAVITQu4vr4xnSDxMaL",
        "JBFqnCBsd6RMkjVDRZzb",
        "pNInz6obpgDQGcFmaJgB",
        "CwhRBWXzGAHq8TQ4Fs17",
        "N2lVS1w4EtoT3dr4eOWO",
        "SAz9YHcvj6GT2YYXdXww",
    ],
    "accented": [
        "FGY2WhTYpPnrIDTdsKH5",
        "SAz9YHcvj6GT2YYXdXww",
        "Xb7hH8MSUJpSbSDYk0k2",
        "CwhRBWXzGAHq8TQ4Fs17",
        "EXAVITQu4vr4xnSDxMaL",
        "N2lVS1w4EtoT3dr4eOWO",
    ],
    "clinical": [
        "IKne3meq5aSn9XLyUdCD",
        "Xb7hH8MSUJpSbSDYk0k2",
        "onwK4e9ZLuTAKqWW03F9",
        "SAz9YHcvj6GT2YYXdXww",
        "FGY2WhTYpPnrIDTdsKH5",
        "CwhRBWXzGAHq8TQ4Fs17",
    ],
}


@dataclass(frozen=True)
class DemoSituationSpec:
    clip_id: str
    base_script_id: str
    situation_label: str
    frontend_slug: str
    legacy_public_relpath: str
    expected_highlights: str
    notes: str


@dataclass(frozen=True)
class DemoVariantSpec:
    variant_id: str
    variant_label: str
    frontend_audio_slug: str
    notes_suffix: str


SITUATION_SPECS = (
    DemoSituationSpec(
        clip_id="demo_20260412_medication_refill",
        base_script_id="medication_refill",
        situation_label="Medication Refill",
        frontend_slug="medication-refill",
        legacy_public_relpath="demo-audio/med-refill.wav",
        expected_highlights="metformin; 500 milligrams; lisinopril; 10 milligrams",
        notes="Patient refill request with dosing and remaining-pill count.",
    ),
    DemoSituationSpec(
        clip_id="demo_20260412_postop_followup",
        base_script_id="postop_followup",
        situation_label="Post-Op Follow-up",
        frontend_slug="post-op-followup",
        legacy_public_relpath="demo-audio/post-op.wav",
        expected_highlights="ibuprofen; 600 milligrams; every 6 hours; every 16 hours",
        notes="Recovery follow-up after knee replacement with interval confusion.",
    ),
    DemoSituationSpec(
        clip_id="demo_20260412_new_symptom_report",
        base_script_id="new_symptom_report",
        situation_label="New Symptom Report",
        frontend_slug="new-symptom-report",
        legacy_public_relpath="demo-audio/symptom-check.wav",
        expected_highlights="amlodipine; 10 milligrams; headaches; dizziness",
        notes="New symptoms after a blood-pressure medication dose change.",
    ),
    DemoSituationSpec(
        clip_id="demo_20260412_allergy_review",
        base_script_id="allergy_review",
        situation_label="Allergy Review",
        frontend_slug="allergy-review",
        legacy_public_relpath="demo-audio/allergy-review.wav",
        expected_highlights="cephalexin; lip swelling; hives; 30 minutes",
        notes="Possible antibiotic allergy review before prescribing.",
    ),
    DemoSituationSpec(
        clip_id="demo_20260412_dose_timing_check",
        base_script_id="dose_timing_check",
        situation_label="Dose Timing Check",
        frontend_slug="dose-timing-check",
        legacy_public_relpath="demo-audio/dose-timing.wav",
        expected_highlights="acetaminophen; ibuprofen; 7 PM; 8 PM; spacing",
        notes="Parent checking whether two medication doses were spaced safely.",
    ),
    DemoSituationSpec(
        clip_id="demo_20260412_rapid_med_list",
        base_script_id="rapid_med_list",
        situation_label="Rapid Med List",
        frontend_slug="rapid-med-list",
        legacy_public_relpath="demo-audio/rapid-meds.wav",
        expected_highlights="metformin; atorvastatin; levothyroxine; warfarin; insulin glargine",
        notes="Fast multi-medication handoff with dense drug and dose content.",
    ),
)

VARIANT_SPECS = (
    DemoVariantSpec(
        variant_id="clear_call",
        variant_label="Clear Call",
        frontend_audio_slug="clear-call",
        notes_suffix="Low-noise baseline telephony.",
    ),
    DemoVariantSpec(
        variant_id="ambient_noise",
        variant_label="Ambient Crowd + TV",
        frontend_audio_slug="ambient-crowd-tv",
        notes_suffix="Rich ambient bed with conversation, room tone, and TV/music texture.",
    ),
    DemoVariantSpec(
        variant_id="heavy_accent",
        variant_label="Heavy Accent",
        frontend_audio_slug="heavy-accent",
        notes_suffix="Strong accented delivery without the ambient noise bed.",
    ),
    DemoVariantSpec(
        variant_id="clinical_handoff",
        variant_label="Clinical Handoff",
        frontend_audio_slug="clinical-handoff",
        notes_suffix="Clinical dictation or staff-to-staff handoff with ambient room bed.",
    ),
)


def build_demo_audio(
    *,
    base_input_path: Path = DEFAULT_BASE_INPUT_PATH,
    out_dir: Path = DEFAULT_OUT_DIR,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    scripts_dir: Path = DEFAULT_SCRIPTS_DIR,
    backend_audio_dir: Path = DEFAULT_BACKEND_AUDIO_DIR,
    frontend_public_dir: Path = DEFAULT_FRONTEND_PUBLIC_DIR,
    concurrency: int = 3,
    model_id: str = "eleven_multilingual_v2",
    timeout_s: float = 120.0,
    resume: bool = False,
) -> dict[str, Any]:
    """Run generation, remix rich noise, export shipped assets, and rewrite the demo manifest."""

    load_audio_gen_env()
    api_key = resolve_elevenlabs_api_key()
    if not api_key:
        raise SystemExit(
            "ELEVENLABS_API_KEY or ELEVEN_LABS_API_KEY is required in environment or backend/.env"
        )

    base_rows = load_base_demo_rows(base_input_path=base_input_path, scripts_dir=scripts_dir)
    variant_rows = build_demo_variant_rows(base_rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    generated_input_path = out_dir / "demo_cards_variants.csv"
    write_generation_input(input_path=generated_input_path, rows=variant_rows)

    config = GenerationConfig(
        input_path=generated_input_path.resolve(),
        out_dir=out_dir.resolve(),
        concurrency=concurrency,
        model_id=model_id,
        resume=resume,
        timeout_s=timeout_s,
    )
    client = ElevenLabsClient(api_key=api_key, timeout_s=timeout_s)
    run_metadata = run_generation(config, client)
    rich_noise_summary = remix_run_dir(run_dir=out_dir, timeout_s=max(timeout_s, 180.0))

    generated_rows = load_generated_clip_rows(out_dir / "clips_rich_noise.jsonl")
    reset_demo_exports(
        backend_audio_dir=backend_audio_dir,
        frontend_public_dir=frontend_public_dir,
    )
    export_rows = export_demo_audio(
        rows=variant_rows,
        generated_rows=generated_rows,
        out_dir=out_dir,
        backend_audio_dir=backend_audio_dir,
        frontend_public_dir=frontend_public_dir,
    )
    write_legacy_aliases(
        generated_rows=export_rows,
        frontend_public_dir=frontend_public_dir,
    )

    manifest_rows = build_demo_manifest_rows()
    write_demo_manifest(manifest_path=manifest_path, rows=manifest_rows)
    validate_demo_manifest(manifest_path=manifest_path, frontend_public_dir=frontend_public_dir)

    return {
        "base_input_path": str(base_input_path.resolve()),
        "generated_input_path": str(generated_input_path.resolve()),
        "out_dir": str(out_dir.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "generated_clip_count": len(variant_rows),
        "exported_backend_audio": [row["backend_audio_path"] for row in export_rows],
        "exported_frontend_audio": [row["frontend_public_path"] for row in export_rows],
        "run_metadata": run_metadata,
        "rich_noise_summary": rich_noise_summary,
    }


def load_base_demo_rows(*, base_input_path: Path, scripts_dir: Path) -> list[dict[str, Any]]:
    """Load and validate the six base demo situations against checked-in scripts."""

    rows = validate_rows(load_input_rows(base_input_path))
    spec_map = {spec.clip_id: spec for spec in SITUATION_SPECS}
    row_map = {str(row["clip_id"]): row for row in rows}

    if len(rows) != len(SITUATION_SPECS):
        raise ValueError(
            f"Base demo input must contain exactly {len(SITUATION_SPECS)} rows; found {len(rows)}"
        )

    if set(row_map) != set(spec_map):
        missing = sorted(set(spec_map) - set(row_map))
        extra = sorted(set(row_map) - set(spec_map))
        raise ValueError(
            f"Base demo clip ids do not match expected set (missing={missing}, extra={extra})"
        )

    ordered_rows: list[dict[str, Any]] = []
    for spec in SITUATION_SPECS:
        row = row_map[spec.clip_id]
        script_path = scripts_dir / f"{spec.base_script_id}.txt"
        if not script_path.exists():
            raise FileNotFoundError(f"Missing demo script: '{script_path}'")
        if str(row["base_script_id"]).strip() != spec.base_script_id:
            raise ValueError(
                f"Row '{spec.clip_id}' must use base_script_id='{spec.base_script_id}'"
            )
        script_text = script_path.read_text(encoding="utf-8").strip()
        if script_text != str(row["text"]).strip():
            raise ValueError(
                f"Base demo CSV text for '{spec.clip_id}' does not match '{script_path.relative_to(REPO_ROOT)}'"
            )
        ordered_rows.append(row)

    return ordered_rows


def build_demo_variant_rows(base_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expand six base situations into four demo-ready variants each."""

    rows: list[dict[str, Any]] = []
    spec_map = {spec.clip_id: spec for spec in SITUATION_SPECS}

    for idx, base_row in enumerate(base_rows):
        situation_spec = spec_map[str(base_row["clip_id"])]
        for variant in VARIANT_SPECS:
            row = dict(base_row)
            row.update(_variant_payload(base_row=base_row, variant=variant, situation_index=idx))
            row["clip_id"] = f"{situation_spec.clip_id}_{variant.variant_id}"
            row["script_family_id"] = situation_spec.clip_id
            rows.append(row)

    validate_rows(rows)
    return rows


def _variant_payload(
    *,
    base_row: dict[str, Any],
    variant: DemoVariantSpec,
    situation_index: int,
) -> dict[str, Any]:
    base_noise_profile = str(base_row.get("noise_profile", "clean")).strip().lower()
    base_accent = str(base_row.get("accent", "us")).strip()
    base_speech_style = str(base_row.get("speech_style", "calm")).strip().lower()
    is_rapid = str(base_row.get("base_script_id", "")).strip() == "rapid_med_list"
    needs_stressed_delivery = str(base_row.get("base_script_id", "")).strip() in {
        "rapid_med_list",
        "dose_timing_check",
    }

    if variant.variant_id == "clear_call":
        return {
            "voice_id": _voice_for_variant("clean", situation_index),
            "voice_type": "neutral",
            "speech_style": "rapid" if is_rapid else "calm",
            "accent": "us",
            "category": "medication_followup",
            "difficulty": "easy",
            "split": "test",
            "noise_level": "low",
            "has_interruptions": False,
            "scenario": "clean_speech",
            "scenario_group": "baseline",
            "noise_profile": "clean",
            "accent_profile": "",
            "medical_domain": False,
            "medical_subtype": "",
        }

    if variant.variant_id == "ambient_noise":
        ambient_noise_profile = base_noise_profile if base_noise_profile in {"medium", "high"} else "medium"
        ambient_noise_level = ambient_noise_profile
        return {
            "voice_id": _voice_for_variant("ambient", situation_index),
            "voice_type": "telephony",
            "speech_style": "rapid" if is_rapid else "conversational",
            "accent": "us",
            "category": "urgent_symptom" if ambient_noise_profile == "high" else "triage_call",
            "difficulty": "hard" if ambient_noise_profile == "high" else "medium",
            "split": "test",
            "noise_level": ambient_noise_level,
            "has_interruptions": True,
            "scenario": "noisy_environment",
            "scenario_group": "noisy",
            "noise_profile": ambient_noise_profile,
            "accent_profile": "",
            "medical_domain": False,
            "medical_subtype": "",
        }

    if variant.variant_id == "heavy_accent":
        accent = base_accent if base_accent and base_accent != "us" else "indian_english"
        speech_style = "rapid" if is_rapid else ("stressed" if needs_stressed_delivery else "careful")
        return {
            "voice_id": _voice_for_variant("accented", situation_index),
            "voice_type": "accented",
            "speech_style": speech_style,
            "accent": accent,
            "category": "medication_question",
            "difficulty": "medium",
            "split": "test",
            "noise_level": "low",
            "has_interruptions": False,
            "scenario": "accented_speech",
            "scenario_group": "accented",
            "noise_profile": "clean",
            "accent_profile": "south_asian_english",
            "medical_domain": False,
            "medical_subtype": "",
        }

    return {
        "voice_id": _voice_for_variant("clinical", situation_index),
        "voice_type": "clinical",
        "speech_style": "rapid" if is_rapid else ("clinical" if base_speech_style != "rapid" else "rapid"),
        "accent": "us",
        "category": "medical_conversation",
        "difficulty": "hard",
        "split": "test",
        "noise_level": "medium",
        "has_interruptions": True,
        "scenario": "medical_conversation",
        "scenario_group": "medical",
        "noise_profile": "medium",
        "accent_profile": "",
        "medical_domain": True,
        "medical_subtype": "medication_safety",
    }


def _voice_for_variant(variant_kind: str, situation_index: int) -> str:
    voices = VOICE_ROTATIONS[variant_kind]
    return voices[situation_index % len(voices)]


def write_generation_input(*, input_path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Write the expanded generation CSV that feeds the normal dataset pipeline."""

    if not rows:
        raise ValueError("No demo variant rows to write")
    input_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_generated_clip_rows(manifest_path: Path) -> dict[str, dict[str, Any]]:
    """Read generated clip rows keyed by clip id."""

    if not manifest_path.exists():
        raise FileNotFoundError(f"Generated clips manifest not found: '{manifest_path}'")

    rows: dict[str, dict[str, Any]] = {}
    for raw in manifest_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        payload = json.loads(line)
        rows[str(payload["clip_id"])] = payload
    return rows


def export_demo_audio(
    *,
    rows: Sequence[dict[str, Any]],
    generated_rows: dict[str, dict[str, Any]],
    out_dir: Path,
    backend_audio_dir: Path,
    frontend_public_dir: Path,
) -> list[dict[str, str]]:
    """Export telephony WAVs into shipped backend/frontend demo locations."""

    backend_audio_dir.mkdir(parents=True, exist_ok=True)
    frontend_audio_dir = frontend_public_dir / "demo-audio"
    frontend_audio_dir.mkdir(parents=True, exist_ok=True)
    situation_map = {spec.clip_id: spec for spec in SITUATION_SPECS}
    variant_map = {spec.variant_id: spec for spec in VARIANT_SPECS}

    exported_rows: list[dict[str, str]] = []
    for row in rows:
        clip_id = str(row["clip_id"])
        generated_row = generated_rows.get(clip_id)
        if not generated_row:
            raise FileNotFoundError(f"Generated clip row missing for '{clip_id}'")

        telephony_relpath = str(generated_row.get("audio_telephony_path", "")).strip()
        if not telephony_relpath:
            raise ValueError(f"Generated clip row for '{clip_id}' is missing audio_telephony_path")

        situation_id, variant_id = split_clip_variant_id(clip_id)
        situation = situation_map[situation_id]
        variant = variant_map[variant_id]
        source_path = out_dir / telephony_relpath
        if not source_path.exists():
            candidate = DEFAULT_OUT_DIR / telephony_relpath
            if not candidate.exists():
                raise FileNotFoundError(f"Generated telephony audio missing: '{candidate}'")
            source_path = candidate.resolve()
        else:
            source_path = source_path.resolve()

        backend_target = backend_audio_dir / f"{clip_id}_take01.wav"
        frontend_target = frontend_public_dir / situation.legacy_public_relpath
        frontend_variant_target = frontend_audio_dir / situation.frontend_slug / f"{variant.frontend_audio_slug}.wav"
        backend_target.parent.mkdir(parents=True, exist_ok=True)
        frontend_variant_target.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(source_path, backend_target)
        shutil.copy2(source_path, frontend_variant_target)

        exported_rows.append(
            {
                "clip_id": clip_id,
                "situation_id": situation_id,
                "variant_id": variant_id,
                "source_path": str(source_path.resolve()),
                "backend_audio_path": str(backend_target.resolve()),
                "frontend_public_path": str(frontend_variant_target.resolve()),
                "legacy_public_path": str(frontend_target.resolve()),
            }
        )

    return exported_rows


def reset_demo_exports(*, backend_audio_dir: Path, frontend_public_dir: Path) -> None:
    """Remove previously shipped demo WAVs so removed scenarios do not linger."""

    backend_audio_dir.mkdir(parents=True, exist_ok=True)
    for wav_path in backend_audio_dir.glob("*.wav"):
        wav_path.unlink()

    frontend_audio_dir = frontend_public_dir / "demo-audio"
    frontend_audio_dir.mkdir(parents=True, exist_ok=True)
    for wav_path in frontend_audio_dir.glob("*.wav"):
        wav_path.unlink()
    for child in frontend_audio_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)


def write_legacy_aliases(*, generated_rows: Sequence[dict[str, str]], frontend_public_dir: Path) -> None:
    """Preserve the original top-level demo WAV paths as aliases to signature variants."""

    signature_map = {
        "demo_20260412_medication_refill": "clear_call",
        "demo_20260412_postop_followup": "ambient_noise",
        "demo_20260412_new_symptom_report": "clear_call",
        "demo_20260412_allergy_review": "clear_call",
        "demo_20260412_dose_timing_check": "ambient_noise",
        "demo_20260412_rapid_med_list": "clinical_handoff",
    }

    situation_map = {spec.clip_id: spec for spec in SITUATION_SPECS}
    by_key = {
        (row["situation_id"], row["variant_id"]): row
        for row in generated_rows
    }

    for situation_id, variant_id in signature_map.items():
        row = by_key[(situation_id, variant_id)]
        legacy_target = frontend_public_dir / situation_map[situation_id].legacy_public_relpath
        legacy_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(row["frontend_public_path"]), legacy_target)


def split_clip_variant_id(clip_id: str) -> tuple[str, str]:
    """Split a generated clip id into its base situation id and variant id."""

    for variant in VARIANT_SPECS:
        suffix = f"_{variant.variant_id}"
        if clip_id.endswith(suffix):
            return clip_id[: -len(suffix)], variant.variant_id
    raise ValueError(f"Clip id does not end with a known variant suffix: '{clip_id}'")


def build_demo_manifest_rows() -> list[dict[str, str]]:
    """Render manifest rows for the shipped multi-variant demo set."""

    rows: list[dict[str, str]] = []
    for situation in SITUATION_SPECS:
        for variant in VARIANT_SPECS:
            clip_id = f"{situation.clip_id}_{variant.variant_id}"
            rows.append(
                {
                    "demo_id": clip_id,
                    "situation_id": situation.clip_id,
                    "situation": situation.situation_label,
                    "variant_id": variant.variant_id,
                    "variant_label": variant.variant_label,
                    "audio_relpath": f"audio/{clip_id}_take01.wav",
                    "frontend_public_relpath": f"demo-audio/{situation.frontend_slug}/{variant.frontend_audio_slug}.wav",
                    "script_ref": f"scripts/{situation.base_script_id}.txt",
                    "expected_highlights": situation.expected_highlights,
                    "notes": f"{situation.notes} {variant.notes_suffix}",
                }
            )
    return rows


def write_demo_manifest(*, manifest_path: Path, rows: Sequence[dict[str, str]]) -> None:
    """Write the demo manifest with stable column ordering."""

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def validate_demo_manifest(*, manifest_path: Path, frontend_public_dir: Path) -> list[dict[str, str]]:
    """Validate checked-in demo assets against the canonical multi-variant spec."""

    if not manifest_path.exists():
        raise FileNotFoundError(f"Demo manifest not found: '{manifest_path}'")

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    expected_rows = build_demo_manifest_rows()
    if len(rows) != len(expected_rows):
        raise ValueError(
            f"Demo manifest must contain exactly {len(expected_rows)} rows; found {len(rows)}"
        )

    by_id = {row["demo_id"]: row for row in rows}
    for expected in expected_rows:
        row = by_id.get(expected["demo_id"])
        if row != expected:
            raise ValueError(f"Manifest row mismatch for '{expected['demo_id']}'")

        script_path = manifest_path.parent / row["script_ref"]
        if not script_path.exists():
            raise FileNotFoundError(f"Missing script reference in demo manifest: '{script_path}'")

        backend_audio_path = manifest_path.parent / row["audio_relpath"]
        if not backend_audio_path.exists():
            raise FileNotFoundError(f"Missing backend demo audio referenced by manifest: '{backend_audio_path}'")

        frontend_audio_path = frontend_public_dir / row["frontend_public_relpath"]
        if not frontend_audio_path.exists():
            raise FileNotFoundError(
                f"Missing frontend demo audio referenced by manifest: '{frontend_audio_path}'"
            )

    return rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and export the multi-variant demo audio catalog")
    parser.add_argument("--base-input", default=str(DEFAULT_BASE_INPUT_PATH), help="Base six-row demo CSV input")
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help="Output directory for generated demo rows and raw/clean/telephony audio",
    )
    parser.add_argument(
        "--manifest-path",
        default=str(DEFAULT_MANIFEST_PATH),
        help="Path to backend/test_audio/demo/manifest.csv",
    )
    parser.add_argument(
        "--scripts-dir",
        default=str(DEFAULT_SCRIPTS_DIR),
        help="Directory containing the six checked-in demo scripts",
    )
    parser.add_argument(
        "--backend-audio-dir",
        default=str(DEFAULT_BACKEND_AUDIO_DIR),
        help="Destination directory for backend demo WAVs",
    )
    parser.add_argument(
        "--frontend-public-dir",
        default=str(DEFAULT_FRONTEND_PUBLIC_DIR),
        help="Path to frontend/public (demo WAVs are exported under demo-audio/)",
    )
    parser.add_argument("--concurrency", type=int, default=3, help="Max concurrent generations")
    parser.add_argument("--model-id", default="eleven_multilingual_v2", help="ElevenLabs model id")
    parser.add_argument("--timeout-s", type=float, default=120.0, help="Request/ffmpeg timeout")
    parser.add_argument("--resume", dest="resume", action="store_true", default=False)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_demo_audio(
        base_input_path=Path(args.base_input).expanduser().resolve(),
        out_dir=Path(args.out_dir).expanduser().resolve(),
        manifest_path=Path(args.manifest_path).expanduser().resolve(),
        scripts_dir=Path(args.scripts_dir).expanduser().resolve(),
        backend_audio_dir=Path(args.backend_audio_dir).expanduser().resolve(),
        frontend_public_dir=Path(args.frontend_public_dir).expanduser().resolve(),
        concurrency=args.concurrency,
        model_id=args.model_id,
        timeout_s=args.timeout_s,
        resume=args.resume,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
