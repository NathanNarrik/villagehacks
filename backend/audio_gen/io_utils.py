"""Input/output helpers for audio generation pipeline."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from .constants import (
    ALLOWED_NUMERIC_CONFUSION_TYPES,
    ALLOWED_SCENARIOS,
    ALLOWED_SCENARIO_GROUPS,
    ALLOWED_VOICE_TYPES,
    BOOLEAN_FIELDS,
    MANIFEST_VERSION,
    MEDICAL_TEMPLATE_COLUMNS,
    NUMERIC_TEMPLATE_COLUMNS,
    OUTPUT_CLIPS_FILE,
    OUTPUT_ERRORS_FILE,
    OUTPUT_RUN_METADATA_FILE,
    REQUIRED_COLUMNS,
    REQUIRED_NON_EMPTY_STRING_FIELDS,
    WORD_TEMPLATE_COLUMNS,
)
from .errors import InputValidationError, ResumeGuardError


def compute_input_hash(input_path: Path) -> str:
    """Compute sha256 hash for source metadata file."""

    digest = hashlib.sha256()
    with input_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 64), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_input_rows(input_path: Path) -> list[dict[str, Any]]:
    """Load rows from CSV or JSONL input."""

    if not input_path.exists():
        raise InputValidationError(f"Input file does not exist: '{input_path}'")

    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        rows = _load_csv(input_path)
    elif suffix in {".jsonl", ".ndjson"}:
        rows = _load_jsonl(input_path)
    else:
        raise InputValidationError("Input file must be .csv or .jsonl")

    if not rows:
        raise InputValidationError("Input file contains no rows.")

    return rows


def validate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate rows and normalize boolean fields."""

    discovered_columns = set().union(*(row.keys() for row in rows))
    missing_columns = sorted(REQUIRED_COLUMNS - discovered_columns)
    if missing_columns:
        raise InputValidationError(
            f"Input metadata missing required columns: {', '.join(missing_columns)}"
        )

    seen_clip_ids: set[str] = set()
    normalized_rows: list[dict[str, Any]] = []

    for idx, row in enumerate(rows, start=1):
        normalized = {k: _normalize_value(v) for k, v in row.items()}
        normalized["voice_type"] = str(normalized.get("voice_type", "")).strip().lower()
        normalized["scenario"] = str(normalized.get("scenario", "")).strip().lower()
        normalized["scenario_group"] = str(normalized.get("scenario_group", "")).strip().lower()
        normalized["noise_profile"] = str(normalized.get("noise_profile", "")).strip().lower()
        normalized["numeric_confusion_type"] = str(
            normalized.get("numeric_confusion_type", "")
        ).strip().lower()
        normalized["category"] = str(normalized.get("category", "")).strip().lower()

        for field in REQUIRED_NON_EMPTY_STRING_FIELDS:
            value = normalized.get(field)
            if not isinstance(value, str) or not value.strip():
                raise InputValidationError(
                    f"Row {idx}: required field '{field}' must be a non-empty string"
                )

        for field in BOOLEAN_FIELDS:
            value = normalized.get(field)
            normalized[field] = _parse_bool(value, field_name=field, row_idx=idx)

        clip_id = str(normalized["clip_id"])
        if clip_id in seen_clip_ids:
            raise InputValidationError(f"Row {idx}: duplicate clip_id '{clip_id}'")
        seen_clip_ids.add(clip_id)

        _validate_enum_fields(normalized, row_idx=idx)
        _validate_scenario_rules(normalized, row_idx=idx)

        normalized_rows.append(normalized)

    return normalized_rows


def _validate_scenario_rules(row: dict[str, Any], *, row_idx: int) -> None:
    scenario = str(row.get("scenario", "")).strip()
    scenario_group = str(row.get("scenario_group", "")).strip()
    noise_profile = str(row.get("noise_profile", "")).strip().lower()
    accent_profile = str(row.get("accent_profile", "")).strip()
    medical_subtype = str(row.get("medical_subtype", "")).strip()
    medical_domain = bool(row.get("medical_domain", False))
    category = str(row.get("category", "")).strip().lower()
    numeric_confusion_type = str(row.get("numeric_confusion_type", "")).strip().lower()
    contains_numeric_confusion = bool(row.get("contains_numeric_confusion", False))

    if scenario not in ALLOWED_SCENARIOS:
        allowed = ", ".join(sorted(ALLOWED_SCENARIOS))
        raise InputValidationError(
            f"Row {row_idx}: scenario '{scenario}' is invalid. Allowed: {allowed}"
        )

    expected_group = {
        "clean_speech": "baseline",
        "noisy_environment": "noisy",
        "accented_speech": "accented",
        "medical_conversation": "medical",
    }[scenario]
    if scenario_group != expected_group:
        raise InputValidationError(
            f"Row {row_idx}: scenario '{scenario}' requires scenario_group='{expected_group}'"
        )

    if scenario == "clean_speech" and noise_profile != "clean":
        raise InputValidationError(
            f"Row {row_idx}: clean_speech requires noise_profile='clean'"
        )

    if scenario == "noisy_environment" and noise_profile not in {"medium", "high"}:
        raise InputValidationError(
            f"Row {row_idx}: noisy_environment requires noise_profile in {{'medium', 'high'}}"
        )

    if scenario == "accented_speech" and not accent_profile:
        raise InputValidationError(
            f"Row {row_idx}: accented_speech requires non-empty accent_profile"
        )

    if scenario == "medical_conversation":
        if not medical_domain:
            raise InputValidationError(
                f"Row {row_idx}: medical_conversation requires medical_domain=true"
            )
        if not medical_subtype:
            raise InputValidationError(
                f"Row {row_idx}: medical_conversation requires non-empty medical_subtype"
            )

    if contains_numeric_confusion and numeric_confusion_type == "none":
        raise InputValidationError(
            f"Row {row_idx}: contains_numeric_confusion=true requires typed confusion value"
        )
    if (not contains_numeric_confusion) and numeric_confusion_type != "none":
        raise InputValidationError(
            f"Row {row_idx}: numeric_confusion_type must be 'none' when confusion flag is false"
        )

    medical_categories = {"medical_conversation", "clinical_triage", "adverse_event_followup"}
    if medical_domain and category not in medical_categories:
        allowed = ", ".join(sorted(medical_categories))
        raise InputValidationError(
            f"Row {row_idx}: medical_domain=true requires category in {{{allowed}}}"
        )
    if (not medical_domain) and category in medical_categories:
        raise InputValidationError(
            f"Row {row_idx}: medical_domain=false is inconsistent with category='{category}'"
        )


def _validate_enum_fields(row: dict[str, Any], *, row_idx: int) -> None:
    _require_allowed_value(
        row_idx=row_idx,
        field_name="voice_type",
        value=str(row.get("voice_type", "")),
        allowed_values=ALLOWED_VOICE_TYPES,
    )
    _require_allowed_value(
        row_idx=row_idx,
        field_name="scenario_group",
        value=str(row.get("scenario_group", "")),
        allowed_values=ALLOWED_SCENARIO_GROUPS,
    )
    _require_allowed_value(
        row_idx=row_idx,
        field_name="numeric_confusion_type",
        value=str(row.get("numeric_confusion_type", "")),
        allowed_values=ALLOWED_NUMERIC_CONFUSION_TYPES,
    )


def _require_allowed_value(
    *,
    row_idx: int,
    field_name: str,
    value: str,
    allowed_values: set[str],
) -> None:
    if value in allowed_values:
        return
    allowed = ", ".join(sorted(allowed_values))
    raise InputValidationError(
        f"Row {row_idx}: field '{field_name}' has invalid value '{value}'. Allowed: {allowed}"
    )


def validate_resume_guard(
    *,
    out_dir: Path,
    input_hash: str,
    resume: bool,
) -> None:
    """Enforce deterministic resume with manifest version + input hash."""

    run_metadata_path = out_dir / OUTPUT_RUN_METADATA_FILE
    if not run_metadata_path.exists() or not resume:
        return

    try:
        metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ResumeGuardError(
            f"Could not parse existing run metadata at '{run_metadata_path}'"
        ) from exc

    previous_version = metadata.get("manifest_version")
    previous_hash = metadata.get("input_hash")

    if previous_version != MANIFEST_VERSION:
        raise ResumeGuardError(
            "Resume blocked: manifest_version mismatch "
            f"(existing={previous_version!r}, current={MANIFEST_VERSION!r})"
        )

    if previous_hash != input_hash:
        raise ResumeGuardError(
            "Resume blocked: input_hash mismatch "
            f"(existing={previous_hash!r}, current={input_hash!r})"
        )


def load_successful_clip_ids(*, out_dir: Path, input_hash: str) -> set[str]:
    """Read existing successful clips from clips.jsonl for resumable runs."""

    clips_path = out_dir / OUTPUT_CLIPS_FILE
    if not clips_path.exists():
        return set()

    ids: set[str] = set()
    with clips_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue

            if (
                payload.get("manifest_version") == MANIFEST_VERSION
                and payload.get("input_hash") == input_hash
                and payload.get("clip_id")
            ):
                ids.add(str(payload["clip_id"]))

    return ids


def write_run_metadata(*, out_dir: Path, payload: dict[str, Any]) -> None:
    """Write run metadata json."""

    path = out_dir / OUTPUT_RUN_METADATA_FILE
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_template_files(*, out_dir: Path) -> None:
    """Write versioned template schema files for deferred tables."""

    templates = {
        "word_features.template.jsonl": WORD_TEMPLATE_COLUMNS,
        "numeric_features.template.jsonl": NUMERIC_TEMPLATE_COLUMNS,
        "medical_entities.template.jsonl": MEDICAL_TEMPLATE_COLUMNS,
    }

    for filename, columns in templates.items():
        path = out_dir / filename
        schema_record = {
            "record_type": "schema",
            "schema_version": MANIFEST_VERSION,
            "columns": columns,
        }
        path.write_text(json.dumps(schema_record) + "\n", encoding="utf-8")


def output_files_for_run(*, out_dir: Path, resume: bool) -> tuple[Path, Path, str]:
    """Return clips path, errors path and write mode for JSONL outputs."""

    clips_path = out_dir / OUTPUT_CLIPS_FILE
    errors_path = out_dir / OUTPUT_ERRORS_FILE
    mode = "a" if resume else "w"

    if not resume:
        clips_path.write_text("", encoding="utf-8")
        errors_path.write_text("", encoding="utf-8")

    return clips_path, errors_path, mode


def _load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise InputValidationError("CSV input is missing a header row.")
        return [dict(row) for row in reader]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as exc:
                raise InputValidationError(f"Invalid JSONL at line {idx}") from exc
            if not isinstance(parsed, dict):
                raise InputValidationError(f"JSONL line {idx} must be an object")
            rows.append(parsed)
    return rows


def _normalize_value(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip()
    return value


def _parse_bool(value: Any, *, field_name: str, row_idx: int) -> bool:
    if isinstance(value, bool):
        return value

    if value is None:
        raise InputValidationError(f"Row {row_idx}: boolean field '{field_name}' is missing")

    text = str(value).strip().lower()
    truthy = {"true", "1", "yes", "y"}
    falsy = {"false", "0", "no", "n"}

    if text in truthy:
        return True
    if text in falsy:
        return False

    raise InputValidationError(
        f"Row {row_idx}: boolean field '{field_name}' has invalid value {value!r}"
    )
