"""Build expanded clip metadata CSV from a separator-delimited text file."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

VOICE_IDS = [
    "CwhRBWXzGAHq8TQ4Fs17",
    "EXAVITQu4vr4xnSDxMaL",
    "FGY2WhTYpPnrIDTdsKH5",
    "IKne3meq5aSn9XLyUdCD",
    "JBFqnCBsd6RMkjVDRZzb",
    "N2lVS1w4EtoT3dr4eOWO",
    "SAz9YHcvj6GT2YYXdXww",
    "Xb7hH8MSUJpSbSDYk0k2",
    "onwK4e9ZLuTAKqWW03F9",
    "pNInz6obpgDQGcFmaJgB",
]

PROFILES = [
    {
        "variant_suffix": "clean",
        "scenario": "clean_speech",
        "scenario_group": "baseline",
        "noise_profile": "clean",
        "noise_level": "low",
        "has_interruptions": False,
        "voice_type": "neutral",
        "speech_style": "calm",
        "accent": "us",
        "accent_profile": "",
        "category": "medication_followup",
        "difficulty": "easy",
        "split": "train",
        "medical_domain": False,
        "medical_subtype": "",
    },
    {
        "variant_suffix": "noisy_med",
        "scenario": "noisy_environment",
        "scenario_group": "noisy",
        "noise_profile": "medium",
        "noise_level": "medium",
        "has_interruptions": True,
        "voice_type": "telephony",
        "speech_style": "conversational",
        "accent": "us",
        "accent_profile": "",
        "category": "triage_call",
        "difficulty": "medium",
        "split": "train",
        "medical_domain": False,
        "medical_subtype": "",
    },
    {
        "variant_suffix": "noisy_high",
        "scenario": "noisy_environment",
        "scenario_group": "noisy",
        "noise_profile": "high",
        "noise_level": "high",
        "has_interruptions": True,
        "voice_type": "telephony",
        "speech_style": "stressed",
        "accent": "us",
        "accent_profile": "",
        "category": "urgent_symptom",
        "difficulty": "hard",
        "split": "val",
        "medical_domain": False,
        "medical_subtype": "",
    },
    {
        "variant_suffix": "accented",
        "scenario": "accented_speech",
        "scenario_group": "accented",
        "noise_profile": "clean",
        "noise_level": "low",
        "has_interruptions": False,
        "voice_type": "accented",
        "speech_style": "careful",
        "accent": "indian_english",
        "accent_profile": "south_asian_english",
        "category": "medication_question",
        "difficulty": "medium",
        "split": "train",
        "medical_domain": False,
        "medical_subtype": "",
    },
    {
        "variant_suffix": "medical",
        "scenario": "medical_conversation",
        "scenario_group": "medical",
        "noise_profile": "medium",
        "noise_level": "medium",
        "has_interruptions": True,
        "voice_type": "clinical",
        "speech_style": "clinical",
        "accent": "us",
        "accent_profile": "",
        "category": "medical_conversation",
        "difficulty": "hard",
        "split": "test",
        "medical_domain": True,
        "medical_subtype": "medication_safety",
    },
]

OUTPUT_COLUMNS = [
    "clip_id",
    "script_family_id",
    "base_script_id",
    "text",
    "voice_id",
    "voice_type",
    "speech_style",
    "accent",
    "category",
    "difficulty",
    "split",
    "noise_level",
    "has_interruptions",
    "contains_numeric_confusion",
    "numeric_confusion_type",
    "contains_medical_terms",
    "contains_ambiguity",
    "scenario",
    "scenario_group",
    "noise_profile",
    "accent_profile",
    "medical_domain",
    "medical_subtype",
]

MEDICAL_TERMS = {
    "metformin",
    "lisinopril",
    "amlodipine",
    "atorvastatin",
    "metoprolol",
    "oxycodone",
    "hydrocodone",
    "hydroxyzine",
    "ibuprofen",
    "acetaminophen",
    "tramadol",
    "prednisone",
    "prednisolone",
    "amoxicillin",
    "penicillin",
    "sulfa",
    "methotrexate",
    "warfarin",
    "levothyroxine",
    "insulin",
    "glipizide",
}

AMBIGUITY_MARKERS = (
    "maybe",
    "not sure",
    "i think",
    "might be",
    "or maybe",
    "i can\'t remember",
    "i cant remember",
    "i'm not totally sure",
    "im not totally sure",
)

NUMERIC_PATTERN = re.compile(r"\b\d+\b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 5x variant clip metadata CSV")
    parser.add_argument(
        "--clips-file",
        default="backend/audio_gen/clips.txt",
        help="Path to source clips text file. '---' lines are treated as separators.",
    )
    parser.add_argument(
        "--output",
        default="backend/audio_gen/input/clips_5x_variants.csv",
        help="Output CSV path",
    )
    return parser.parse_args()


def load_clips(path: Path) -> list[str]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if line and line != "---"]


def derive_numeric_confusion_type(text: str) -> str:
    lowered = text.lower()
    has_numbers = bool(NUMERIC_PATTERN.search(lowered))
    has_ambiguity = any(marker in lowered for marker in AMBIGUITY_MARKERS)
    if not has_numbers or not has_ambiguity:
        return "none"

    duration_hints = (
        "hour",
        "hours",
        "day",
        "days",
        "week",
        "weeks",
        "minute",
        "minutes",
        "every",
        "daily",
        "twice",
        "once",
    )
    dose_hints = (
        "mg",
        "milligram",
        "milligrams",
        "tablet",
        "tablets",
        "dose",
        "doses",
    )

    if any(token in lowered for token in duration_hints):
        return "duration_confusion"
    if any(token in lowered for token in dose_hints):
        return "dose_confusion"
    return "digit_vs_digit"


def build_rows(clips: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    for base_idx, text in enumerate(clips, start=1):
        text_lower = text.lower()
        numeric_confusion_type = derive_numeric_confusion_type(text)
        contains_numeric_confusion = numeric_confusion_type != "none"
        contains_medical_terms = any(term in text_lower for term in MEDICAL_TERMS)
        contains_ambiguity = any(marker in text_lower for marker in AMBIGUITY_MARKERS)

        for variant_idx, profile in enumerate(PROFILES, start=1):
            voice_id = VOICE_IDS[(base_idx + variant_idx - 2) % len(VOICE_IDS)]
            clip_id = f"clip_{base_idx:04d}_{profile['variant_suffix']}"

            rows.append(
                {
                    "clip_id": clip_id,
                    "script_family_id": f"family_{base_idx:04d}",
                    "base_script_id": f"script_{base_idx:04d}",
                    "text": text,
                    "voice_id": voice_id,
                    "voice_type": profile["voice_type"],
                    "speech_style": profile["speech_style"],
                    "accent": profile["accent"],
                    "category": profile["category"],
                    "difficulty": profile["difficulty"],
                    "split": profile["split"],
                    "noise_level": profile["noise_level"],
                    "has_interruptions": "true" if profile["has_interruptions"] else "false",
                    "contains_numeric_confusion": "true"
                    if contains_numeric_confusion
                    else "false",
                    "numeric_confusion_type": numeric_confusion_type,
                    "contains_medical_terms": "true" if contains_medical_terms else "false",
                    "contains_ambiguity": "true" if contains_ambiguity else "false",
                    "scenario": profile["scenario"],
                    "scenario_group": profile["scenario_group"],
                    "noise_profile": profile["noise_profile"],
                    "accent_profile": profile["accent_profile"],
                    "medical_domain": "true" if profile["medical_domain"] else "false",
                    "medical_subtype": profile["medical_subtype"],
                }
            )

    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    clips_file = Path(args.clips_file)
    output_file = Path(args.output)

    clips = load_clips(clips_file)
    rows = build_rows(clips)
    write_csv(output_file, rows)

    print(f"Clips source: {clips_file}")
    print(f"Base clips parsed (excluding separators): {len(clips)}")
    print(f"Variants per clip: {len(PROFILES)}")
    print(f"Total rows written: {len(rows)}")
    print(f"Output CSV: {output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
