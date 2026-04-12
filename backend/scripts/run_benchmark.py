"""Generate benchmark_results.json for the demo backend.

Default behavior remains safe for hackathon demos:
- If no eval dataset is present, copy baseline benchmark JSON to the output path.
- If eval data is present, recompute per-clip metrics from transcript text.

Expected eval JSONL fields (per row):
- clip_id
- ground_truth (or reference_text/text)
- raw_text (or raw_transcript)
- corrected_text (or corrected_transcript)
Optional fields:
- category
- difficulty (Standard/Adversarial)
- medical_keywords (list[str] or comma-delimited str)
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Sequence

BACKEND_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_PATH = BACKEND_DIR / "data" / "benchmark_results.json"
DEFAULT_RESULTS_EXAMPLE_PATH = BACKEND_DIR / "data" / "benchmark_results.json.example"
DEFAULT_EVAL_PATH = BACKEND_DIR / "data" / "benchmark_eval.jsonl"
DEFAULT_MANIFEST_PATH = BACKEND_DIR / "test_audio" / "benchmark" / "v1" / "manifest.csv"

_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
_DIGIT_RE = re.compile(r"\d")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate backend/data/benchmark_results.json")
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Baseline benchmark JSON to copy/augment (default: existing benchmark_results.json else .example)",
    )
    parser.add_argument(
        "--eval",
        type=Path,
        default=DEFAULT_EVAL_PATH,
        help="JSONL eval file with ground-truth/raw/corrected transcript text",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        help="Output benchmark JSON path",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Benchmark manifest CSV (ground_truth/keywords/category/difficulty metadata)",
    )
    return parser.parse_args()


def _resolve_source_path(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    if DEFAULT_RESULTS_PATH.exists():
        return DEFAULT_RESULTS_PATH
    return DEFAULT_RESULTS_EXAMPLE_PATH


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Benchmark source not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Malformed benchmark source JSON ({path}): {exc}") from exc


def _load_manifest_rows(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            out: dict[str, dict[str, str]] = {}
            for row in reader:
                clip_id = str((row or {}).get("clip_id") or "").strip()
                if not clip_id:
                    continue
                out[clip_id] = {str(k): str(v or "").strip() for k, v in row.items()}
            return out
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Malformed benchmark manifest CSV ({path}): {exc}") from exc


def _round(value: float) -> float:
    return round(float(value), 4)


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return _round(sum(values) / len(values))


def _mean_optional(values: Sequence[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    if not present:
        return None
    return _mean(present)


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if isinstance(item, dict):
                tok = item.get("word") or item.get("text")
                if tok:
                    out.append(str(tok))
            elif item is not None:
                out.append(str(item))
        return " ".join(out)
    if isinstance(value, dict):
        text = value.get("text")
        return str(text) if text is not None else ""
    return str(value)


def _tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _digits(text: str) -> list[str]:
    return _DIGIT_RE.findall(text)


def _levenshtein(a: Sequence[str], b: Sequence[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ai in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, bj in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (0 if ai == bj else 1)
            cur[j] = min(ins, delete, sub)
        prev = cur
    return prev[-1]


def _error_rate_ratio(reference: Sequence[str], hypothesis: Sequence[str]) -> float:
    if not reference:
        return 0.0
    return _levenshtein(reference, hypothesis) / len(reference)


def _accuracy_ratio(reference: Sequence[str], hypothesis: Sequence[str]) -> float | None:
    if not reference:
        return None
    dist = _levenshtein(reference, hypothesis)
    value = 1.0 - (dist / len(reference))
    return max(0.0, min(1.0, value))


def _contains_phrase(tokens: Sequence[str], phrase: Sequence[str]) -> bool:
    size = len(phrase)
    if size == 0 or len(tokens) < size:
        return False
    for idx in range(0, len(tokens) - size + 1):
        if list(tokens[idx : idx + size]) == list(phrase):
            return True
    return False


def _load_medical_terms() -> list[list[str]]:
    try:
        from app.keyterms import load_initial_keyterms  # type: ignore

        terms = load_initial_keyterms()
    except Exception:
        terms = [
            "metformin",
            "lisinopril",
            "atorvastatin",
            "amoxicillin",
            "ibuprofen",
            "acetaminophen",
            "warfarin",
            "insulin",
        ]

    tokenized = []
    for term in terms:
        toks = _tokens(str(term))
        if toks:
            tokenized.append(toks)
    return tokenized


def _parse_medical_keywords(value: Any) -> list[list[str]]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
    elif isinstance(value, list):
        parts = [str(v).strip() for v in value if str(v).strip()]
    else:
        parts = [str(value).strip()]

    out: list[list[str]] = []
    for item in parts:
        toks = _tokens(item)
        if toks:
            out.append(toks)
    return out


def _medical_keyword_accuracy_ratio(
    reference_text: str,
    hypothesis_text: str,
    medical_terms: list[list[str]],
    explicit_keywords: list[list[str]],
) -> float | None:
    ref_tokens = _tokens(reference_text)
    hyp_tokens = _tokens(hypothesis_text)

    reference_terms = explicit_keywords or [
        term for term in medical_terms if _contains_phrase(ref_tokens, term)
    ]
    if not reference_terms:
        return None

    matches = 0
    for term in reference_terms:
        if _contains_phrase(hyp_tokens, term):
            matches += 1

    return matches / len(reference_terms)


def _scale_from_source(payload: dict[str, Any]) -> tuple[float, float]:
    """Return (wer_scale, rate_scale).

    Scale is either 1.0 (ratio style, 0..1) or 100.0 (percent style, 0..100).
    """

    wer_scale = 1.0
    for row in payload.get("results", []):
        raw = row.get("raw_wer")
        if isinstance(raw, (int, float)) and raw > 1.0:
            wer_scale = 100.0
            break

    rate_scale = 1.0
    metrics = payload.get("metrics", {})
    verification_rate = metrics.get("verification_rate") if isinstance(metrics, dict) else None
    if isinstance(verification_rate, (int, float)) and verification_rate > 1.0:
        rate_scale = 100.0

    return wer_scale, rate_scale


def _scaled_optional(value: float | None, scale: float) -> float | None:
    if value is None:
        return None
    return _round(value * scale)


def _compute_result_row(
    row: dict[str, Any],
    source_rows_by_clip: dict[str, dict[str, Any]],
    manifest_rows_by_clip: dict[str, dict[str, str]],
    medical_terms: list[list[str]],
    wer_scale: float,
    rate_scale: float,
) -> dict[str, Any]:
    clip_id = str(row.get("clip_id", "")).strip()
    if not clip_id:
        raise ValueError("eval row missing clip_id")

    source = source_rows_by_clip.get(clip_id, {})
    manifest = manifest_rows_by_clip.get(clip_id, {})

    reference_text = _coerce_text(
        row.get("ground_truth")
        or row.get("reference_text")
        or row.get("text")
        or manifest.get("ground_truth")
    )
    raw_text = _coerce_text(row.get("raw_text") or row.get("raw_transcript"))
    corrected_text = _coerce_text(
        row.get("corrected_text") or row.get("corrected_transcript")
    )
    if not reference_text:
        raise ValueError(f"eval row {clip_id} missing ground_truth/reference_text/text")

    raw_word_error = _error_rate_ratio(_tokens(reference_text), _tokens(raw_text))
    corrected_word_error = _error_rate_ratio(_tokens(reference_text), _tokens(corrected_text))

    raw_char_error = _error_rate_ratio(list("".join(_tokens(reference_text))), list("".join(_tokens(raw_text))))
    corrected_char_error = _error_rate_ratio(
        list("".join(_tokens(reference_text))),
        list("".join(_tokens(corrected_text))),
    )

    raw_digit_acc = _accuracy_ratio(_digits(reference_text), _digits(raw_text))
    corrected_digit_acc = _accuracy_ratio(_digits(reference_text), _digits(corrected_text))

    explicit_keywords = _parse_medical_keywords(
        row.get("medical_keywords") or manifest.get("medical_keywords")
    )
    raw_medical_acc = _medical_keyword_accuracy_ratio(
        reference_text,
        raw_text,
        medical_terms,
        explicit_keywords,
    )
    corrected_medical_acc = _medical_keyword_accuracy_ratio(
        reference_text,
        corrected_text,
        medical_terms,
        explicit_keywords,
    )

    improvement_pct = 0.0
    if raw_word_error > 0:
        improvement_pct = max(0.0, ((raw_word_error - corrected_word_error) / raw_word_error) * 100.0)

    difficulty_raw = str(
        row.get("difficulty")
        or source.get("difficulty")
        or manifest.get("difficulty")
        or "Standard"
    ).strip()
    difficulty = "Adversarial" if difficulty_raw.lower().startswith("ad") else "Standard"

    return {
        "clip_id": clip_id,
        "category": str(
            row.get("category")
            or source.get("category")
            or manifest.get("category")
            or "Unknown"
        ),
        "difficulty": difficulty,
        "raw_wer": _round(raw_word_error * wer_scale),
        "corrected_wer": _round(corrected_word_error * wer_scale),
        "raw_cer": _round(raw_char_error * wer_scale),
        "corrected_cer": _round(corrected_char_error * wer_scale),
        "raw_digit_accuracy": _scaled_optional(raw_digit_acc, rate_scale),
        "corrected_digit_accuracy": _scaled_optional(corrected_digit_acc, rate_scale),
        "raw_medical_keyword_accuracy": _scaled_optional(raw_medical_acc, rate_scale),
        "corrected_medical_keyword_accuracy": _scaled_optional(corrected_medical_acc, rate_scale),
        "improvement_pct": _round(improvement_pct),
    }


def _add_optional_fields(payload: dict[str, Any]) -> dict[str, Any]:
    for row in payload.get("results", []):
        if isinstance(row, dict):
            row.setdefault("raw_cer", None)
            row.setdefault("corrected_cer", None)
            row.setdefault("raw_digit_accuracy", None)
            row.setdefault("corrected_digit_accuracy", None)
            row.setdefault("raw_medical_keyword_accuracy", None)
            row.setdefault("corrected_medical_keyword_accuracy", None)

    aggregate = payload.setdefault("aggregate", {})
    if isinstance(aggregate, dict):
        aggregate.setdefault("avg_raw_cer", None)
        aggregate.setdefault("avg_corrected_cer", None)
        aggregate.setdefault("avg_raw_digit_accuracy", None)
        aggregate.setdefault("avg_corrected_digit_accuracy", None)
        aggregate.setdefault("avg_raw_medical_keyword_accuracy", None)
        aggregate.setdefault("avg_corrected_medical_keyword_accuracy", None)

    metrics = payload.setdefault("metrics", {})
    if isinstance(metrics, dict):
        metrics.setdefault("digit_accuracy_coverage", None)
        metrics.setdefault("medical_keyword_accuracy_coverage", None)

    return payload


def _apply_manifest_metadata(
    payload: dict[str, Any],
    manifest_rows_by_clip: dict[str, dict[str, str]],
) -> dict[str, Any]:
    if not manifest_rows_by_clip:
        return payload

    for row in payload.get("results", []):
        if not isinstance(row, dict):
            continue
        clip_id = str(row.get("clip_id") or "").strip()
        if not clip_id:
            continue
        manifest = manifest_rows_by_clip.get(clip_id)
        if not manifest:
            continue

        manifest_category = str(manifest.get("category") or "").strip()
        if manifest_category:
            row["category"] = manifest_category

        manifest_difficulty = str(manifest.get("difficulty") or "").strip()
        if manifest_difficulty:
            row["difficulty"] = (
                "Adversarial"
                if manifest_difficulty.lower().startswith("ad")
                else "Standard"
            )

    return payload


def _recompute(
    payload: dict[str, Any],
    eval_path: Path,
    manifest_rows_by_clip: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    manifest_rows_by_clip = manifest_rows_by_clip or {}
    payload = _apply_manifest_metadata(payload, manifest_rows_by_clip)
    if not eval_path.exists():
        return _add_optional_fields(payload)

    lines = [line.strip() for line in eval_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return _add_optional_fields(payload)

    source_rows = payload.get("results", [])
    source_rows_by_clip = {
        str(item.get("clip_id")): item
        for item in source_rows
        if isinstance(item, dict) and item.get("clip_id")
    }

    wer_scale, rate_scale = _scale_from_source(payload)
    medical_terms = _load_medical_terms()

    computed_results: list[dict[str, Any]] = []
    for idx, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Malformed eval JSONL at line {idx}: {exc}") from exc
        if not isinstance(row, dict):
            raise SystemExit(f"Malformed eval JSONL at line {idx}: expected object")
        try:
            computed_results.append(
                _compute_result_row(
                    row=row,
                    source_rows_by_clip=source_rows_by_clip,
                    manifest_rows_by_clip=manifest_rows_by_clip,
                    medical_terms=medical_terms,
                    wer_scale=wer_scale,
                    rate_scale=rate_scale,
                )
            )
        except ValueError as exc:
            raise SystemExit(f"Invalid eval row at line {idx}: {exc}") from exc

    aggregate = payload.setdefault("aggregate", {})
    if not isinstance(aggregate, dict):
        aggregate = {}

    aggregate.update(
        {
            "avg_raw_wer": _mean([r["raw_wer"] for r in computed_results]),
            "avg_corrected_wer": _mean([r["corrected_wer"] for r in computed_results]),
            "avg_raw_cer": _mean([r["raw_cer"] for r in computed_results]),
            "avg_corrected_cer": _mean([r["corrected_cer"] for r in computed_results]),
            "avg_improvement_pct": _mean([r["improvement_pct"] for r in computed_results]),
            "avg_raw_digit_accuracy": _mean_optional(
                [r.get("raw_digit_accuracy") for r in computed_results]
            ),
            "avg_corrected_digit_accuracy": _mean_optional(
                [r.get("corrected_digit_accuracy") for r in computed_results]
            ),
            "avg_raw_medical_keyword_accuracy": _mean_optional(
                [r.get("raw_medical_keyword_accuracy") for r in computed_results]
            ),
            "avg_corrected_medical_keyword_accuracy": _mean_optional(
                [r.get("corrected_medical_keyword_accuracy") for r in computed_results]
            ),
        }
    )

    total = len(computed_results)
    raw_digit_count = sum(1 for r in computed_results if r.get("raw_digit_accuracy") is not None)
    raw_medical_count = sum(
        1 for r in computed_results if r.get("raw_medical_keyword_accuracy") is not None
    )
    metrics = payload.setdefault("metrics", {})
    if isinstance(metrics, dict):
        digit_coverage = (raw_digit_count / total) if total else None
        medical_coverage = (raw_medical_count / total) if total else None
        metrics["digit_accuracy_coverage"] = _scaled_optional(digit_coverage, rate_scale)
        metrics["medical_keyword_accuracy_coverage"] = _scaled_optional(
            medical_coverage, rate_scale
        )

    payload["results"] = computed_results
    payload["aggregate"] = aggregate
    return payload


def main() -> int:
    args = _parse_args()
    source = _resolve_source_path(args.source)
    payload = _load_json(source)
    manifest_rows_by_clip = _load_manifest_rows(args.manifest)
    payload = _recompute(payload, args.eval, manifest_rows_by_clip=manifest_rows_by_clip)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.eval.exists():
        print(f"Wrote recomputed benchmark artifact from eval rows: {args.out}")
    else:
        print(f"Wrote benchmark artifact from baseline source: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
