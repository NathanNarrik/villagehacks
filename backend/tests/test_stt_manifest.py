from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from backend.stt.build_telephony_manifest import (
    build_staged_dataset,
    load_examples,
)


def _write_source_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _source_row(
    *,
    clip_id: str,
    split: str,
    text: str = "example text",
    audio_relpath: str | None = None,
    accent: str = "us",
    accent_profile: str = "",
) -> dict[str, str]:
    return {
        "clip_id": clip_id,
        "audio_telephony_path": audio_relpath or f"telephony/{clip_id}.wav",
        "text": text,
        "split": split,
        "accent": accent,
        "accent_profile": accent_profile,
    }


def test_load_examples_reads_telephony_path_and_accent_profile(tmp_path: Path) -> None:
    source_csv = tmp_path / "source.csv"
    telephony_dir = tmp_path / "telephony"
    telephony_dir.mkdir(parents=True, exist_ok=True)
    (telephony_dir / "clip_1.wav").write_bytes(b"wav")

    _write_source_csv(
        source_csv,
        [
            _source_row(
                clip_id="clip_1",
                split="train",
                accent="indian_english",
                accent_profile="south_asian_english",
            )
        ],
    )

    examples = load_examples(source_csv)

    assert len(examples) == 1
    assert examples[0].source_audio_path == (telephony_dir / "clip_1.wav").resolve()
    assert examples[0].accent_bucket == "south_asian_english"


def test_build_staged_dataset_balances_train_accents_and_copies_audio(tmp_path: Path) -> None:
    source_csv = tmp_path / "source.csv"
    telephony_dir = tmp_path / "telephony"
    telephony_dir.mkdir(parents=True, exist_ok=True)
    for clip_id in ("clip_us_1", "clip_us_2", "clip_us_3", "clip_sa_1", "clip_val_1"):
        (telephony_dir / f"{clip_id}.wav").write_bytes(b"wav")

    _write_source_csv(
        source_csv,
        [
            _source_row(clip_id="clip_us_1", split="train", accent="us"),
            _source_row(clip_id="clip_us_2", split="train", accent="us"),
            _source_row(clip_id="clip_us_3", split="train", accent="us"),
            _source_row(
                clip_id="clip_sa_1",
                split="train",
                accent="indian_english",
                accent_profile="south_asian_english",
            ),
            _source_row(clip_id="clip_val_1", split="val", accent="us"),
        ],
    )

    output_dir = tmp_path / "out"
    manifest_path, summary_path = build_staged_dataset(
        source_csv=source_csv,
        output_dir=output_dir,
        copy_audio=True,
        balance_train_accents_enabled=True,
        target_ratio=0.5,
        min_distinct_train_accents=2,
        min_train_samples_per_accent=1,
    )

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    train_rows = [row for row in manifest_rows if row["split"] == "train"]
    south_asian_count = sum(1 for row in train_rows if row["text"] == "example text")
    assert len(train_rows) == 5
    assert south_asian_count == 5
    assert all(row["audio_path"].startswith("audio/") for row in manifest_rows)
    assert (output_dir / "audio" / "clip_us_1.wav").exists()
    assert summary["split_accent_counts"]["train"]["south_asian_english"] == 2
    assert summary["split_accent_counts"]["train"]["us"] == 3


def test_build_staged_dataset_requires_multiple_train_accents(tmp_path: Path) -> None:
    source_csv = tmp_path / "source.csv"
    telephony_dir = tmp_path / "telephony"
    telephony_dir.mkdir(parents=True, exist_ok=True)
    for clip_id in ("clip_1", "clip_2"):
        (telephony_dir / f"{clip_id}.wav").write_bytes(b"wav")

    _write_source_csv(
        source_csv,
        [
            _source_row(clip_id="clip_1", split="train", accent="us"),
            _source_row(clip_id="clip_2", split="train", accent="us"),
        ],
    )

    with pytest.raises(ValueError, match="enough accent buckets"):
        build_staged_dataset(
            source_csv=source_csv,
            output_dir=tmp_path / "out",
            copy_audio=False,
            balance_train_accents_enabled=False,
            min_distinct_train_accents=2,
            min_train_samples_per_accent=1,
        )
