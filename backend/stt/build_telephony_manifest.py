"""Build a Colab-ready telephony manifest from generated audio assets.

This script stages the telephony dataset for Whisper fine-tuning by:
1. Reading the generated telephony metadata CSV.
2. Selecting `audio_telephony_path`, `text`, and `split`.
3. Optionally oversampling underrepresented train accents.
4. Optionally copying referenced audio into one Colab-friendly folder.

The resulting folder is designed to be copied to Google Drive as:

    /content/drive/MyDrive/carecaller/
      telephony_manifest.csv
      telephony_manifest_summary.json
      audio/*.wav
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path

VALID_SPLITS = {"train", "val", "test"}
PACKAGE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = PACKAGE_DIR.parent
DEFAULT_SOURCE_CSV = BACKEND_DIR / "audio_gen" / "output" / "run_5x_v2" / "clips_rich_noise_rebalanced.csv"
DEFAULT_OUTPUT_DIR = PACKAGE_DIR / "colab_dataset"
DEFAULT_SUMMARY_NAME = "telephony_manifest_summary.json"
DEFAULT_MANIFEST_NAME = "telephony_manifest.csv"


@dataclass(slots=True)
class TelephonyExample:
    """One telephony training example."""

    clip_id: str
    split: str
    text: str
    accent_bucket: str
    source_audio_path: Path
    copy_index: int = 0


def _normalize_accent_bucket(row: dict[str, str]) -> str:
    accent_profile = str(row.get("accent_profile") or "").strip().lower()
    accent = str(row.get("accent") or "").strip().lower()
    if accent_profile:
        return accent_profile
    if accent:
        return accent
    return "unknown"


def load_examples(source_csv: Path) -> list[TelephonyExample]:
    """Load telephony examples from the generated metadata CSV."""
    if not source_csv.exists():
        raise FileNotFoundError(f"Source CSV not found: {source_csv}")

    required_columns = {"clip_id", "audio_telephony_path", "text", "split"}
    examples: list[TelephonyExample] = []
    source_root = source_csv.parent

    with source_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = sorted(required_columns - fieldnames)
        if missing:
            raise ValueError(
                f"Source CSV is missing required columns: {', '.join(missing)}"
            )

        for row in reader:
            clip_id = str(row.get("clip_id") or "").strip()
            split = str(row.get("split") or "").strip().lower()
            text = str(row.get("text") or "").strip()
            rel_audio_path = str(row.get("audio_telephony_path") or "").strip()
            if not clip_id or not rel_audio_path or not text:
                continue
            if split not in VALID_SPLITS:
                raise ValueError(
                    f"Unexpected split {split!r} for clip {clip_id!r}; expected one of {sorted(VALID_SPLITS)}"
                )

            audio_path = (source_root / rel_audio_path).resolve()
            if not audio_path.exists():
                raise FileNotFoundError(
                    f"Telephony audio path for {clip_id!r} does not exist: {audio_path}"
                )

            examples.append(
                TelephonyExample(
                    clip_id=clip_id,
                    split=split,
                    text=text,
                    accent_bucket=_normalize_accent_bucket(row),
                    source_audio_path=audio_path,
                )
            )

    if not examples:
        raise ValueError(f"No usable rows were found in {source_csv}")
    return examples


def _split_accent_counts(examples: list[TelephonyExample]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for example in examples:
        counts[example.split][example.accent_bucket] += 1
    return {
        split: dict(sorted(counter.items()))
        for split, counter in sorted(counts.items())
    }


def _validate_accent_coverage(
    examples: list[TelephonyExample],
    *,
    min_distinct_train_accents: int,
    min_train_samples_per_accent: int,
) -> None:
    train_counts = Counter(
        example.accent_bucket for example in examples if example.split == "train"
    )
    if len(train_counts) < min_distinct_train_accents:
        raise ValueError(
            "Training split does not contain enough accent buckets: "
            f"found {len(train_counts)}, require at least {min_distinct_train_accents}. "
            f"Buckets present: {dict(sorted(train_counts.items()))}"
        )
    too_small = {
        accent: count
        for accent, count in train_counts.items()
        if count < min_train_samples_per_accent
    }
    if too_small:
        raise ValueError(
            "Training split is too thin for one or more accent buckets: "
            f"{too_small}. Increase data or lower --min-train-samples-per-accent."
        )


def balance_train_accents(
    examples: list[TelephonyExample],
    *,
    target_ratio: float,
    min_train_samples_per_accent: int,
) -> list[TelephonyExample]:
    """Oversample underrepresented train accents by duplicating manifest rows."""
    if target_ratio <= 0:
        return list(examples)

    balanced = list(examples)
    train_groups: dict[str, list[TelephonyExample]] = defaultdict(list)
    for example in examples:
        if example.split == "train":
            train_groups[example.accent_bucket].append(example)

    if not train_groups:
        return balanced

    max_count = max(len(group) for group in train_groups.values())
    target_count = max(
        min_train_samples_per_accent,
        int(math.ceil(max_count * target_ratio)),
    )

    for accent, group in sorted(train_groups.items()):
        needed = max(0, target_count - len(group))
        for offset in range(needed):
            seed = group[offset % len(group)]
            balanced.append(replace(seed, copy_index=offset + 1))

    return balanced


def write_staged_dataset(
    examples: list[TelephonyExample],
    *,
    output_dir: Path,
    source_csv: Path,
    copy_audio: bool,
    balance_train_accents_enabled: bool,
    target_ratio: float,
    min_distinct_train_accents: int,
    min_train_samples_per_accent: int,
) -> tuple[Path, Path]:
    """Write the manifest and accent summary to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / DEFAULT_MANIFEST_NAME
    summary_path = output_dir / DEFAULT_SUMMARY_NAME

    copied_audio_paths: dict[Path, str] = {}
    audio_dir = output_dir / "audio"
    if copy_audio:
        audio_dir.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["audio_path", "text", "split"])
        writer.writeheader()
        for example in examples:
            if copy_audio:
                manifest_audio_path = copied_audio_paths.get(example.source_audio_path)
                if manifest_audio_path is None:
                    destination = audio_dir / example.source_audio_path.name
                    if not destination.exists():
                        shutil.copy2(example.source_audio_path, destination)
                    manifest_audio_path = str(Path("audio") / destination.name)
                    copied_audio_paths[example.source_audio_path] = manifest_audio_path
            else:
                manifest_audio_path = str(example.source_audio_path)

            writer.writerow(
                {
                    "audio_path": manifest_audio_path,
                    "text": example.text,
                    "split": example.split,
                }
            )

    train_counts = Counter(
        example.accent_bucket for example in examples if example.split == "train"
    )
    summary_payload = {
        "source_csv": str(source_csv),
        "manifest_path": str(manifest_path),
        "copy_audio": copy_audio,
        "balanced_train_accents": balance_train_accents_enabled,
        "balance_policy": {
            "target_ratio": target_ratio,
            "min_distinct_train_accents": min_distinct_train_accents,
            "min_train_samples_per_accent": min_train_samples_per_accent,
        },
        "total_rows": len(examples),
        "split_counts": dict(sorted(Counter(example.split for example in examples).items())),
        "split_accent_counts": _split_accent_counts(examples),
        "train_accent_share": {
            accent: round(count / max(sum(train_counts.values()), 1), 4)
            for accent, count in sorted(train_counts.items())
        },
    }
    summary_path.write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return manifest_path, summary_path


def build_staged_dataset(
    *,
    source_csv: Path = DEFAULT_SOURCE_CSV,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    copy_audio: bool = True,
    balance_train_accents_enabled: bool = True,
    target_ratio: float = 0.5,
    min_distinct_train_accents: int = 2,
    min_train_samples_per_accent: int = 40,
) -> tuple[Path, Path]:
    """Build a Colab-ready telephony manifest and optional staged audio folder."""
    base_examples = load_examples(source_csv)
    final_examples = list(base_examples)
    if balance_train_accents_enabled:
        final_examples = balance_train_accents(
            final_examples,
            target_ratio=target_ratio,
            min_train_samples_per_accent=min_train_samples_per_accent,
        )

    _validate_accent_coverage(
        final_examples,
        min_distinct_train_accents=min_distinct_train_accents,
        min_train_samples_per_accent=min_train_samples_per_accent,
    )

    return write_staged_dataset(
        final_examples,
        output_dir=output_dir,
        source_csv=source_csv,
        copy_audio=copy_audio,
        balance_train_accents_enabled=balance_train_accents_enabled,
        target_ratio=target_ratio,
        min_distinct_train_accents=min_distinct_train_accents,
        min_train_samples_per_accent=min_train_samples_per_accent,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage a Colab-ready telephony manifest for Whisper fine-tuning."
    )
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--copy-audio", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--balance-train-accents",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--train-accent-target-ratio", type=float, default=0.5)
    parser.add_argument("--min-distinct-train-accents", type=int, default=2)
    parser.add_argument("--min-train-samples-per-accent", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest_path, summary_path = build_staged_dataset(
        source_csv=args.source_csv,
        output_dir=args.output_dir,
        copy_audio=bool(args.copy_audio),
        balance_train_accents_enabled=bool(args.balance_train_accents),
        target_ratio=float(args.train_accent_target_ratio),
        min_distinct_train_accents=int(args.min_distinct_train_accents),
        min_train_samples_per_accent=int(args.min_train_samples_per_accent),
    )
    print(f"manifest={manifest_path}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
