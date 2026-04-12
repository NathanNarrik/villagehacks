from __future__ import annotations

from backend.audio_gen.build_variants import (
    ACCENT_VOICE_IDS,
    STANDARD_VOICE_IDS,
    build_rows,
)


def test_build_rows_emits_five_variants_per_clip() -> None:
    rows = build_rows(["I take metformin 500 mg twice daily."])

    assert len(rows) == 5
    assert {row["scenario"] for row in rows} == {
        "clean_speech",
        "noisy_environment",
        "accented_speech",
        "medical_conversation",
    }


def test_accent_profiles_use_accent_voice_pool() -> None:
    rows = build_rows(["I take lisinopril 10 mg daily."])
    by_suffix = {row["clip_id"].split("_")[-1]: row for row in rows}

    assert by_suffix["clean"]["voice_id"] in STANDARD_VOICE_IDS
    assert by_suffix["noisy_med"]["voice_id"] in ACCENT_VOICE_IDS
    assert by_suffix["accented"]["voice_id"] in ACCENT_VOICE_IDS
    assert by_suffix["medical"]["voice_id"] in ACCENT_VOICE_IDS
    assert by_suffix["accented"]["accent_profile"] == "mixed_non_us_english"
    assert by_suffix["noisy_med"]["accent_profile"] == "mixed_non_us_english"
