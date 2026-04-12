from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from app.schemas import ScribeWord
from xgb import infer
from xgb.features import build_word_rows_for_clip
from xgb.learning_loop import load_state, update_from_corrected_call


def _w(text: str, start_ms: int, end_ms: int, confidence: float | None = None) -> ScribeWord:
    return ScribeWord(
        text=text,
        start_ms=start_ms,
        end_ms=end_ms,
        speaker_id="speaker_0",
        confidence=confidence,
    )


def test_build_word_rows_exact_match_has_no_risky_tokens():
    rows = build_word_rows_for_clip(
        clip_id="clip_1",
        words=[_w("metformin", 0, 100), _w("daily", 110, 180)],
        clip_metadata={"scenario": "clean_speech", "noise_profile": "clean", "split": "train"},
        corrected_text="metformin daily",
        keyterms=["metformin"],
        correction_frequency={},
    )

    assert [row.needs_verification for row in rows] == [0, 0]
    assert rows[0].matches_keyterm == 1


def test_build_word_rows_flags_medical_and_numeric_confusions():
    rows = build_word_rows_for_clip(
        clip_id="clip_2",
        words=[_w("metoformin", 0, 100), _w("16", 120, 170), _w("hours", 180, 260)],
        clip_metadata={"scenario": "noisy_environment", "noise_profile": "high", "split": "train"},
        corrected_text="metformin 6 hours",
        keyterms=["metformin"],
        correction_frequency={"metoformin": 3},
    )

    assert rows[0].needs_verification == 1
    assert rows[1].needs_verification == 1
    assert rows[0].correction_frequency == 3
    assert rows[0].is_medical_candidate == 1
    assert rows[1].is_numeric == 1


def test_build_word_rows_marks_insert_neighbors_and_timing_features():
    rows = build_word_rows_for_clip(
        clip_id="clip_3",
        words=[
            _w("take", 0, 50),
            _w("metformin", 80, 170),
            _w("nightly", 200, 320, confidence=0.2),
        ],
        clip_metadata={"scenario": "clean_speech", "noise_profile": "clean", "has_interruptions": "false"},
        corrected_text="please take metformin nightly",
        keyterms=["metformin"],
        correction_frequency={},
    )

    assert rows[0].needs_verification == 1
    assert rows[0].pause_before_ms == 0
    assert rows[0].pause_after_ms == 30
    assert rows[1].pause_before_ms == 30
    assert rows[2].is_low_confidence_from_stt == 1
    assert len(rows) == 3


def test_learning_loop_updates_state_and_is_idempotent(tmp_path: Path):
    state_path = tmp_path / "learning_state.json"
    dataset_path = tmp_path / "dataset.csv"
    manifest_row = {
        "clip_id": "clip_4",
        "scenario": "medical_conversation",
        "noise_profile": "medium",
        "accent_profile": "unknown",
        "split": "train",
        "has_interruptions": "true",
    }
    scribe_payload = {
        "clip_id": "clip_4",
        "words": [
            {"text": "metoformin", "start_ms": 0, "end_ms": 100, "speaker_id": "speaker_0"},
            {"text": "16", "start_ms": 120, "end_ms": 170, "speaker_id": "speaker_0"},
        ],
    }
    corrected_row = {
        "clip_id": "clip_4",
        "corrected_text": "metformin 6",
    }

    first_rows = update_from_corrected_call(
        clip_id="clip_4",
        manifest_row=manifest_row,
        scribe_payload=scribe_payload,
        corrected_row=corrected_row,
        state_path=state_path,
        dataset_path=dataset_path,
    )
    second_rows = update_from_corrected_call(
        clip_id="clip_4",
        manifest_row=manifest_row,
        scribe_payload=scribe_payload,
        corrected_row=corrected_row,
        state_path=state_path,
        dataset_path=dataset_path,
    )

    assert len(first_rows) == 2
    assert second_rows == []

    state = load_state(state_path)
    assert state.phonetic_map["metoformin"] == "metformin"
    assert state.correction_frequency["metoformin"] == 1
    assert state.numeric_confusion_stats["16->6"] == 1

    with dataset_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert len(rows) == 2


def test_inference_groups_high_risk_spans(monkeypatch: pytest.MonkeyPatch):
    pd = pytest.importorskip("pandas")
    np = pytest.importorskip("numpy")

    class FakePreprocessor:
        def transform(self, frame):
            assert isinstance(frame, pd.DataFrame)
            return frame

    class FakeClassifier:
        def predict_proba(self, frame):
            values = []
            for _, row in frame.iterrows():
                risk = 0.9 if row["word_text"] in {"metoformin", "16"} else 0.1
                values.append([1.0 - risk, risk])
            return np.array(values)

    monkeypatch.setattr(
        infer,
        "_load_bundle",
        lambda path=infer.DEFAULT_XGB_MODEL_PATH: {
            "preprocessor": FakePreprocessor(),
            "classifier": FakeClassifier(),
            "raw_feature_names": [
                "word_text",
                "previous_word",
                "next_word",
                "speaker",
                "noise_profile",
                "accent_profile",
                "scenario",
                "is_numeric",
                "is_medical_candidate",
                "matches_keyterm",
                "phonetic_distance_to_nearest_keyterm",
                "timing_irregularity_score",
                "pause_before_ms",
                "pause_after_ms",
                "word_duration_ms",
                "has_interruptions",
                "is_low_confidence_from_stt",
                "correction_frequency",
            ],
        },
    )

    result = infer.score_transcript_words(
        clip_id="clip_5",
        words=[
            _w("metoformin", 0, 100),
            _w("16", 120, 170),
            _w("hours", 180, 240),
        ],
        clip_metadata={"scenario": "noisy_environment", "noise_profile": "high"},
        keyterms=["metformin"],
        correction_frequency={},
        low_threshold=0.6,
    )

    assert result is not None
    assert len(result.word_scores) == 3
    assert len(result.high_risk_spans) == 1
    assert result.high_risk_spans[0].text == "metoformin 16"


def test_training_and_viz_smoke(tmp_path: Path):
    pytest.importorskip("pandas")
    pytest.importorskip("numpy")
    pytest.importorskip("sklearn")
    pytest.importorskip("xgboost")
    pytest.importorskip("joblib")
    pytest.importorskip("matplotlib")

    from xgb.train import train_model
    from xgb.viz import plot_feature_importance, plot_retraining_snapshots, plot_training_history

    dataset_path = tmp_path / "dataset.csv"
    artifact_path = tmp_path / "model.joblib"
    schema_path = tmp_path / "schema.json"
    history_path = tmp_path / "history.json"
    snapshots_path = tmp_path / "snapshots.csv"

    rows = [
        {
            "clip_id": f"clip_{idx}",
            "word_index": 0,
            "start_ms": 0,
            "end_ms": 100,
            "split": split,
            "word_text": word,
            "previous_word": "",
            "next_word": "daily",
            "speaker": "Doctor",
            "is_numeric": 0,
            "is_medical_candidate": int(label),
            "matches_keyterm": int(label),
            "phonetic_distance_to_nearest_keyterm": 0.1 if label else 0.9,
            "timing_irregularity_score": 0.2,
            "pause_before_ms": 0,
            "pause_after_ms": 20,
            "word_duration_ms": 100,
            "noise_profile": "clean",
            "accent_profile": "unknown",
            "scenario": "clean_speech",
            "has_interruptions": 0,
            "is_low_confidence_from_stt": 0,
            "correction_frequency": 2 if label else 0,
            "needs_verification": label,
        }
        for idx, (word, label, split) in enumerate(
            [
                ("metoformin", 1, "train"),
                ("metformin", 0, "train"),
                ("lisinipril", 1, "train"),
                ("hello", 0, "train"),
                ("prednizone", 1, "val"),
                ("thanks", 0, "val"),
                ("warferin", 1, "test"),
                ("daily", 0, "test"),
            ]
        )
    ]

    with dataset_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    result = train_model(
        dataset_path=dataset_path,
        artifact_path=artifact_path,
        feature_schema_path=schema_path,
        history_path=history_path,
        snapshots_path=snapshots_path,
    )

    assert artifact_path.exists()
    assert schema_path.exists()
    assert history_path.exists()
    assert snapshots_path.exists()
    assert result.row_count == 8

    training_plot = plot_training_history(history_path=history_path, output_path=tmp_path / "history.png")
    snapshots_plot = plot_retraining_snapshots(
        snapshots_path=snapshots_path,
        output_path=tmp_path / "snapshots.png",
    )
    importance_plot = plot_feature_importance(
        model_path=artifact_path,
        output_path=tmp_path / "importance.png",
    )

    assert training_plot.exists()
    assert snapshots_plot.exists()
    assert importance_plot.exists()
