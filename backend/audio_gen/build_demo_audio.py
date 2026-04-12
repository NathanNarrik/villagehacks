"""Generate and export the final six demo audio clips."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .constants import OUTPUT_CLIPS_FILE
from .env_utils import load_audio_gen_env, resolve_elevenlabs_api_key
from .elevenlabs import ElevenLabsClient
from .generator import GenerationConfig, run_generation
from .io_utils import load_input_rows, validate_rows

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = REPO_ROOT / "backend" / "audio_gen" / "input" / "demo_cards_20260412.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "backend" / "audio_gen" / "output" / "demo_cards_20260412"
DEFAULT_MANIFEST_PATH = REPO_ROOT / "backend" / "test_audio" / "demo" / "manifest.csv"
DEFAULT_SCRIPTS_DIR = REPO_ROOT / "backend" / "test_audio" / "demo" / "scripts"
DEFAULT_BACKEND_AUDIO_DIR = REPO_ROOT / "backend" / "test_audio" / "demo" / "audio"
DEFAULT_FRONTEND_PUBLIC_DIR = REPO_ROOT / "frontend" / "public"

MANIFEST_FIELDNAMES = [
    "demo_id",
    "scenario",
    "audio_relpath",
    "frontend_public_relpath",
    "script_ref",
    "expected_highlights",
    "notes",
]


@dataclass(frozen=True)
class DemoAudioSpec:
    clip_id: str
    base_script_id: str
    scenario_label: str
    backend_audio_filename: str
    frontend_public_relpath: str
    expected_highlights: str
    notes: str


DEMO_AUDIO_SPECS = (
    DemoAudioSpec(
        clip_id="demo_20260412_medication_refill",
        base_script_id="medication_refill",
        scenario_label="Medication Refill",
        backend_audio_filename="demo_20260412_medication_refill_take01.wav",
        frontend_public_relpath="demo-audio/med-refill.wav",
        expected_highlights="metformin; 500 milligrams; lisinopril; 10 milligrams",
        notes="Neutral caller, clean telephony",
    ),
    DemoAudioSpec(
        clip_id="demo_20260412_postop_followup",
        base_script_id="postop_followup",
        scenario_label="Post-Op Follow-up",
        backend_audio_filename="demo_20260412_postop_followup_take01.wav",
        frontend_public_relpath="demo-audio/post-op.wav",
        expected_highlights="ibuprofen; 600 milligrams; every 6 hours; every 16 hours",
        notes="Telephony caller with moderate background noise",
    ),
    DemoAudioSpec(
        clip_id="demo_20260412_new_symptom_report",
        base_script_id="new_symptom_report",
        scenario_label="New Symptom Report",
        backend_audio_filename="demo_20260412_new_symptom_report_take01.wav",
        frontend_public_relpath="demo-audio/symptom-check.wav",
        expected_highlights="amlodipine; 10 milligrams; headaches; dizziness",
        notes="Accented speaker, clean telephony",
    ),
    DemoAudioSpec(
        clip_id="demo_20260412_allergy_review",
        base_script_id="allergy_review",
        scenario_label="Allergy Review",
        backend_audio_filename="demo_20260412_allergy_review_take01.wav",
        frontend_public_relpath="demo-audio/allergy-review.wav",
        expected_highlights="cephalexin; lip swelling; hives; 30 minutes",
        notes="Neutral caller reviewing a prior allergy reaction",
    ),
    DemoAudioSpec(
        clip_id="demo_20260412_heavy_accent_noise",
        base_script_id="heavy_accent_noise",
        scenario_label="Heavy Accent + Noise",
        backend_audio_filename="demo_20260412_heavy_accent_noise_take01.wav",
        frontend_public_relpath="demo-audio/adversarial-accent.wav",
        expected_highlights="apixaban; 2.5; 5 milligrams; chest tightness",
        notes="Speakerphone accent with heavy background TV-style noise",
    ),
    DemoAudioSpec(
        clip_id="demo_20260412_rapid_med_list",
        base_script_id="rapid_med_list",
        scenario_label="Rapid Med List",
        backend_audio_filename="demo_20260412_rapid_med_list_take01.wav",
        frontend_public_relpath="demo-audio/rapid-meds.wav",
        expected_highlights="metformin; atorvastatin; levothyroxine; warfarin; insulin glargine",
        notes="Fast clinical dictation with dense medication names and dosing",
    ),
)


def build_demo_audio(
    *,
    input_path: Path = DEFAULT_INPUT_PATH,
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
    """Run generation, export shipped assets, and rewrite the demo manifest."""

    load_audio_gen_env()
    api_key = resolve_elevenlabs_api_key()
    if not api_key:
        raise SystemExit(
            "ELEVENLABS_API_KEY or ELEVEN_LABS_API_KEY is required in environment or backend/.env"
        )

    rows = load_demo_rows(input_path=input_path, scripts_dir=scripts_dir)

    config = GenerationConfig(
        input_path=input_path.resolve(),
        out_dir=out_dir.resolve(),
        concurrency=concurrency,
        model_id=model_id,
        resume=resume,
        timeout_s=timeout_s,
    )
    client = ElevenLabsClient(api_key=api_key, timeout_s=timeout_s)
    run_metadata = run_generation(config, client)

    generated_rows = load_generated_clip_rows(out_dir)
    export_rows = export_demo_audio(
        rows=rows,
        generated_rows=generated_rows,
        out_dir=out_dir,
        backend_audio_dir=backend_audio_dir,
        frontend_public_dir=frontend_public_dir,
    )
    manifest_rows = build_demo_manifest_rows()
    write_demo_manifest(manifest_path=manifest_path, rows=manifest_rows)
    validate_demo_manifest(manifest_path=manifest_path, frontend_public_dir=frontend_public_dir)

    return {
        "input_path": str(input_path.resolve()),
        "out_dir": str(out_dir.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "generated_clip_count": len(rows),
        "exported_backend_audio": [row["backend_audio_path"] for row in export_rows],
        "exported_frontend_audio": [row["frontend_public_path"] for row in export_rows],
        "run_metadata": run_metadata,
    }


def load_demo_rows(*, input_path: Path, scripts_dir: Path) -> list[dict[str, Any]]:
    """Load canonical demo CSV rows and ensure they stay in sync with scripts/specs."""

    rows = validate_rows(load_input_rows(input_path))
    spec_map = {spec.clip_id: spec for spec in DEMO_AUDIO_SPECS}
    row_map = {str(row["clip_id"]): row for row in rows}

    if len(rows) != len(DEMO_AUDIO_SPECS):
        raise ValueError(
            f"Demo input must contain exactly {len(DEMO_AUDIO_SPECS)} rows; found {len(rows)}"
        )

    if set(row_map) != set(spec_map):
        missing = sorted(set(spec_map) - set(row_map))
        extra = sorted(set(row_map) - set(spec_map))
        raise ValueError(f"Demo clip ids do not match expected set (missing={missing}, extra={extra})")

    ordered_rows: list[dict[str, Any]] = []
    for spec in DEMO_AUDIO_SPECS:
        row = row_map[spec.clip_id]
        script_path = scripts_dir / f"{spec.base_script_id}.txt"
        if not script_path.exists():
            raise FileNotFoundError(f"Missing demo script: '{script_path}'")

        if str(row["base_script_id"]).strip() != spec.base_script_id:
            raise ValueError(
                f"Row '{spec.clip_id}' must use base_script_id='{spec.base_script_id}'"
            )

        script_text = script_path.read_text(encoding="utf-8").strip()
        row_text = str(row["text"]).strip()
        if script_text != row_text:
            raise ValueError(
                f"Demo CSV text for '{spec.clip_id}' does not match '{script_path.relative_to(REPO_ROOT)}'"
            )

        ordered_rows.append(row)

    return ordered_rows


def load_generated_clip_rows(out_dir: Path) -> dict[str, dict[str, Any]]:
    """Read generated clip rows keyed by clip id."""

    clips_path = out_dir / OUTPUT_CLIPS_FILE
    if not clips_path.exists():
        raise FileNotFoundError(f"Generated clips manifest not found: '{clips_path}'")

    rows: dict[str, dict[str, Any]] = {}
    for raw in clips_path.read_text(encoding="utf-8").splitlines():
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

    spec_map = {spec.clip_id: spec for spec in DEMO_AUDIO_SPECS}
    exported_rows: list[dict[str, str]] = []

    for row in rows:
        clip_id = str(row["clip_id"])
        spec = spec_map[clip_id]
        generated_row = generated_rows.get(clip_id)
        if not generated_row:
            raise FileNotFoundError(f"Generated clip row missing for '{clip_id}'")

        telephony_relpath = str(generated_row.get("audio_telephony_path", "")).strip()
        if not telephony_relpath:
            raise ValueError(f"Generated clip row for '{clip_id}' is missing audio_telephony_path")

        source_path = out_dir / telephony_relpath
        if not source_path.exists():
            raise FileNotFoundError(f"Generated telephony audio missing: '{source_path}'")

        backend_target = backend_audio_dir / spec.backend_audio_filename
        frontend_target = frontend_public_dir / spec.frontend_public_relpath
        backend_target.parent.mkdir(parents=True, exist_ok=True)
        frontend_target.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(source_path, backend_target)
        shutil.copy2(source_path, frontend_target)

        exported_rows.append(
            {
                "clip_id": clip_id,
                "source_path": str(source_path.resolve()),
                "backend_audio_path": str(backend_target.resolve()),
                "frontend_public_path": str(frontend_target.resolve()),
            }
        )

    return exported_rows


def build_demo_manifest_rows() -> list[dict[str, str]]:
    """Render manifest rows for the shipped six-clip demo set."""

    return [
        {
            "demo_id": spec.clip_id,
            "scenario": spec.scenario_label,
            "audio_relpath": f"audio/{spec.backend_audio_filename}",
            "frontend_public_relpath": spec.frontend_public_relpath,
            "script_ref": f"scripts/{spec.base_script_id}.txt",
            "expected_highlights": spec.expected_highlights,
            "notes": spec.notes,
        }
        for spec in DEMO_AUDIO_SPECS
    ]


def write_demo_manifest(*, manifest_path: Path, rows: Sequence[dict[str, str]]) -> None:
    """Write the demo manifest with stable column ordering."""

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def validate_demo_manifest(*, manifest_path: Path, frontend_public_dir: Path) -> list[dict[str, str]]:
    """Validate checked-in demo assets against the canonical six-row spec."""

    if not manifest_path.exists():
        raise FileNotFoundError(f"Demo manifest not found: '{manifest_path}'")

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if len(rows) != len(DEMO_AUDIO_SPECS):
        raise ValueError(
            f"Demo manifest must contain exactly {len(DEMO_AUDIO_SPECS)} rows; found {len(rows)}"
        )

    approved_frontend_paths = {spec.frontend_public_relpath for spec in DEMO_AUDIO_SPECS}
    spec_map = {spec.clip_id: spec for spec in DEMO_AUDIO_SPECS}
    row_map = {str(row["demo_id"]): row for row in rows}

    if set(row_map) != set(spec_map):
        missing = sorted(set(spec_map) - set(row_map))
        extra = sorted(set(row_map) - set(spec_map))
        raise ValueError(
            f"Demo manifest ids do not match expected set (missing={missing}, extra={extra})"
        )

    for spec in DEMO_AUDIO_SPECS:
        row = row_map[spec.clip_id]
        script_ref = str(row["script_ref"]).strip()
        audio_relpath = str(row["audio_relpath"]).strip()
        frontend_relpath = str(row["frontend_public_relpath"]).strip()

        if str(row["scenario"]).strip() != spec.scenario_label:
            raise ValueError(f"Scenario label mismatch for '{spec.clip_id}'")
        if audio_relpath != f"audio/{spec.backend_audio_filename}":
            raise ValueError(f"Backend audio path mismatch for '{spec.clip_id}'")
        if frontend_relpath != spec.frontend_public_relpath:
            raise ValueError(f"Frontend audio path mismatch for '{spec.clip_id}'")
        if frontend_relpath not in approved_frontend_paths:
            raise ValueError(f"Unapproved frontend audio path in manifest: '{frontend_relpath}'")

        script_path = manifest_path.parent / script_ref
        if not script_path.exists():
            raise FileNotFoundError(f"Missing script reference in demo manifest: '{script_path}'")

        backend_audio_path = manifest_path.parent / audio_relpath
        if not backend_audio_path.exists():
            raise FileNotFoundError(f"Missing backend demo audio referenced by manifest: '{backend_audio_path}'")

        frontend_audio_path = frontend_public_dir / frontend_relpath
        if not frontend_audio_path.exists():
            raise FileNotFoundError(
                f"Missing frontend demo audio referenced by manifest: '{frontend_audio_path}'"
            )

    return rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and export the six canonical demo audios")
    parser.add_argument("--input", default=str(DEFAULT_INPUT_PATH), help="Canonical demo CSV input")
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
        help="Destination directory for dated backend demo WAVs",
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
        input_path=Path(args.input).expanduser().resolve(),
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
