from __future__ import annotations

from pathlib import Path

import pytest

from backend.audio_gen.build_demo_audio import (
    DEMO_AUDIO_SPECS,
    DEFAULT_INPUT_PATH,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_SCRIPTS_DIR,
    build_demo_manifest_rows,
    export_demo_audio,
    load_demo_rows,
    validate_demo_manifest,
)
from backend.audio_gen.env_utils import resolve_elevenlabs_api_key

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({"ELEVENLABS_API_KEY": "primary-key"}, "primary-key"),
        ({"ELEVEN_LABS_API_KEY": "legacy-key"}, "legacy-key"),
        (
            {"ELEVENLABS_API_KEY": "primary-key", "ELEVEN_LABS_API_KEY": "legacy-key"},
            "primary-key",
        ),
    ],
)
def test_resolve_elevenlabs_api_key_accepts_both_names(
    env: dict[str, str], expected: str
) -> None:
    assert resolve_elevenlabs_api_key(env) == expected


def test_export_demo_audio_copies_expected_backend_and_frontend_assets(tmp_path: Path) -> None:
    out_dir = tmp_path / "generated"
    backend_audio_dir = tmp_path / "backend" / "audio"
    frontend_public_dir = tmp_path / "frontend" / "public"

    rows = [{"clip_id": spec.clip_id} for spec in DEMO_AUDIO_SPECS]
    generated_rows: dict[str, dict[str, str]] = {}

    for spec in DEMO_AUDIO_SPECS:
        source_path = out_dir / "telephony" / f"{spec.clip_id}.wav"
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_bytes = spec.clip_id.encode("utf-8")
        source_path.write_bytes(source_bytes)
        generated_rows[spec.clip_id] = {
            "clip_id": spec.clip_id,
            "audio_telephony_path": f"telephony/{spec.clip_id}.wav",
        }

    exported = export_demo_audio(
        rows=rows,
        generated_rows=generated_rows,
        out_dir=out_dir,
        backend_audio_dir=backend_audio_dir,
        frontend_public_dir=frontend_public_dir,
    )

    assert len(exported) == len(DEMO_AUDIO_SPECS)
    for spec in DEMO_AUDIO_SPECS:
        backend_target = backend_audio_dir / spec.backend_audio_filename
        frontend_target = frontend_public_dir / spec.frontend_public_relpath
        assert backend_target.exists()
        assert frontend_target.exists()
        assert backend_target.read_bytes() == spec.clip_id.encode("utf-8")
        assert frontend_target.read_bytes() == spec.clip_id.encode("utf-8")


def test_checked_in_demo_csv_matches_scripts() -> None:
    rows = load_demo_rows(input_path=DEFAULT_INPUT_PATH, scripts_dir=DEFAULT_SCRIPTS_DIR)

    assert [row["clip_id"] for row in rows] == [spec.clip_id for spec in DEMO_AUDIO_SPECS]


def test_checked_in_demo_manifest_matches_scripts_and_frontend_assets() -> None:
    rows = validate_demo_manifest(
        manifest_path=DEFAULT_MANIFEST_PATH,
        frontend_public_dir=REPO_ROOT / "frontend" / "public",
    )

    assert len(rows) == len(DEMO_AUDIO_SPECS)
    assert [row["demo_id"] for row in rows] == [spec.clip_id for spec in DEMO_AUDIO_SPECS]
    assert rows == build_demo_manifest_rows()
