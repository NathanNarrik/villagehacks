"""File-backed learning loop for word-risk modeling."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.medical_patterns import matches_medical, normalize

from .features import (
    DEFAULT_DATASET_PATH,
    DEFAULT_LEARNING_STATE_PATH,
    WordFeatureRow,
    append_dataset_rows,
    build_word_rows_for_clip,
    corrected_text_from_row,
    ensure_directories,
    risky_word_indices,
    scribe_words_from_payload,
    transcript_tokens,
)


@dataclass(slots=True)
class LearningState:
    """Persisted learning state."""

    phonetic_map: dict[str, str] = field(default_factory=dict)
    keyterm_counts: dict[str, int] = field(default_factory=dict)
    correction_frequency: dict[str, int] = field(default_factory=dict)
    numeric_confusion_stats: dict[str, int] = field(default_factory=dict)
    processed_clip_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "phonetic_map": dict(sorted(self.phonetic_map.items())),
            "keyterm_counts": dict(sorted(self.keyterm_counts.items())),
            "correction_frequency": dict(sorted(self.correction_frequency.items())),
            "numeric_confusion_stats": dict(sorted(self.numeric_confusion_stats.items())),
            "processed_clip_ids": sorted(set(self.processed_clip_ids)),
        }


def load_state(path: Path = DEFAULT_LEARNING_STATE_PATH) -> LearningState:
    """Load persisted learning state, or return empty defaults."""
    if not path.exists():
        return LearningState()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return LearningState()
    return LearningState(
        phonetic_map={str(k): str(v) for k, v in dict(payload.get("phonetic_map") or {}).items()},
        keyterm_counts={str(k): int(v) for k, v in dict(payload.get("keyterm_counts") or {}).items()},
        correction_frequency={
            str(k): int(v) for k, v in dict(payload.get("correction_frequency") or {}).items()
        },
        numeric_confusion_stats={
            str(k): int(v) for k, v in dict(payload.get("numeric_confusion_stats") or {}).items()
        },
        processed_clip_ids=[str(item) for item in list(payload.get("processed_clip_ids") or [])],
    )


def save_state(state: LearningState, path: Path = DEFAULT_LEARNING_STATE_PATH) -> None:
    """Persist learning state to JSON."""
    ensure_directories()
    path.write_text(json.dumps(state.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def _apply_alignment_updates(
    *,
    state: LearningState,
    source_tokens: list[str],
    target_tokens: list[str],
) -> None:
    risky, ops = risky_word_indices(source_tokens, target_tokens)
    del risky
    for op in ops:
        if op.tag != "replace":
            continue
        source_slice = source_tokens[op.src_start : op.src_end]
        target_slice = target_tokens[op.dst_start : op.dst_end]
        if len(source_slice) != 1 or len(target_slice) != 1:
            continue
        source_token = normalize(source_slice[0])
        target_token = normalize(target_slice[0])
        if not source_token or not target_token or source_token == target_token:
            continue
        state.phonetic_map[source_token] = target_token
        state.correction_frequency[source_token] = state.correction_frequency.get(source_token, 0) + 1
        if matches_medical(target_token):
            state.keyterm_counts[target_token] = state.keyterm_counts.get(target_token, 0) + 1
        if source_token.replace(".", "", 1).isdigit() and target_token.replace(".", "", 1).isdigit():
            key = f"{source_token}->{target_token}"
            state.numeric_confusion_stats[key] = state.numeric_confusion_stats.get(key, 0) + 1


def update_from_corrected_call(
    *,
    clip_id: str,
    manifest_row: dict[str, Any],
    scribe_payload: dict[str, Any],
    corrected_row: dict[str, Any],
    state_path: Path = DEFAULT_LEARNING_STATE_PATH,
    dataset_path: Path = DEFAULT_DATASET_PATH,
) -> list[WordFeatureRow]:
    """Update learning state and dataset from one corrected call."""
    state = load_state(state_path)
    if clip_id in set(state.processed_clip_ids):
        return []

    corrected_text = corrected_text_from_row(corrected_row)
    source_words = scribe_words_from_payload(scribe_payload)
    source_tokens = [word.text for word in source_words]
    target_tokens = transcript_tokens(corrected_text)

    _apply_alignment_updates(state=state, source_tokens=source_tokens, target_tokens=target_tokens)

    learned_keyterms = set(state.keyterm_counts) | {
        normalize(token) for token in target_tokens if matches_medical(token)
    }
    rows = build_word_rows_for_clip(
        clip_id=clip_id,
        words=source_words,
        clip_metadata=manifest_row,
        corrected_text=corrected_text,
        keyterms=learned_keyterms,
        correction_frequency=state.correction_frequency,
    )

    append_dataset_rows(rows, dataset_path)
    state.processed_clip_ids.append(clip_id)
    save_state(state, state_path)
    return rows
