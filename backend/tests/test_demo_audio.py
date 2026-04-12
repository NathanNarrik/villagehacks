from __future__ import annotations

from pathlib import Path

import pytest

from backend.audio_gen.build_demo_audio import (
    DEFAULT_BASE_INPUT_PATH,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_SCRIPTS_DIR,
    SITUATION_SPECS,
    VARIANT_SPECS,
    build_demo_manifest_rows,
    build_demo_variant_rows,
    export_demo_audio,
    load_base_demo_rows,
    validate_demo_manifest,
    write_legacy_aliases,
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


def test_build_demo_variant_rows_expands_six_situations_to_four_variants_each() -> None:
    base_rows = load_base_demo_rows(
        base_input_path=DEFAULT_BASE_INPUT_PATH,
        scripts_dir=DEFAULT_SCRIPTS_DIR,
    )
    variant_rows = build_demo_variant_rows(base_rows)

    assert len(base_rows) == len(SITUATION_SPECS)
    assert len(variant_rows) == len(SITUATION_SPECS) * len(VARIANT_SPECS)

    expected_ids = {
        f"{situation.clip_id}_{variant.variant_id}"
        for situation in SITUATION_SPECS
        for variant in VARIANT_SPECS
    }
    assert {row["clip_id"] for row in variant_rows} == expected_ids


def test_export_demo_audio_copies_variant_assets_and_aliases(tmp_path: Path) -> None:
    out_dir = tmp_path / "generated"
    backend_audio_dir = tmp_path / "backend" / "audio"
    frontend_public_dir = tmp_path / "frontend" / "public"

    base_rows = load_base_demo_rows(
        base_input_path=DEFAULT_BASE_INPUT_PATH,
        scripts_dir=DEFAULT_SCRIPTS_DIR,
    )
    variant_rows = build_demo_variant_rows(base_rows)

    generated_rows: dict[str, dict[str, str]] = {}
    for row in variant_rows:
        clip_id = str(row["clip_id"])
        source_path = out_dir / "telephony_rich_noisy" / f"{clip_id}.wav"
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_bytes = clip_id.encode("utf-8")
        source_path.write_bytes(source_bytes)
        generated_rows[clip_id] = {
            "clip_id": clip_id,
            "audio_telephony_path": f"telephony_rich_noisy/{clip_id}.wav",
        }

    exported = export_demo_audio(
        rows=variant_rows,
        generated_rows=generated_rows,
        out_dir=out_dir,
        backend_audio_dir=backend_audio_dir,
        frontend_public_dir=frontend_public_dir,
    )
    write_legacy_aliases(generated_rows=exported, frontend_public_dir=frontend_public_dir)

    assert len(exported) == len(variant_rows)
    for row in exported:
        assert Path(row["backend_audio_path"]).exists()
        assert Path(row["frontend_public_path"]).exists()

    assert (frontend_public_dir / "demo-audio" / "med-refill.wav").exists()
    assert (frontend_public_dir / "demo-audio" / "dose-timing.wav").exists()


def test_checked_in_demo_manifest_matches_scripts_and_frontend_assets() -> None:
    rows = validate_demo_manifest(
        manifest_path=DEFAULT_MANIFEST_PATH,
        frontend_public_dir=REPO_ROOT / "frontend" / "public",
    )

    assert len(rows) == len(SITUATION_SPECS) * len(VARIANT_SPECS)
    assert rows == build_demo_manifest_rows()
