"""Read persisted XGBoost learning-loop artifacts for API/UI use."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .features import (
    DEFAULT_RETRAINING_SNAPSHOTS_PATH,
    DEFAULT_TRAINING_HISTORY_PATH,
    DEFAULT_XGB_MODEL_PATH,
)


def load_training_history(path: Path = DEFAULT_TRAINING_HISTORY_PATH) -> tuple[str | None, list[dict[str, float | int | None]]]:
    """Load per-round training and validation metrics."""
    if not path.exists():
        return None, []

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None, []

    train_metrics = payload.get("validation_0")
    val_metrics = payload.get("validation_1")
    if not isinstance(train_metrics, dict) or not train_metrics:
        return None, []

    metric_name = next(iter(train_metrics.keys()))
    train_values = list(train_metrics.get(metric_name) or [])
    val_values = list(val_metrics.get(metric_name) or []) if isinstance(val_metrics, dict) else []

    rows: list[dict[str, float | int | None]] = []
    for idx, train_value in enumerate(train_values):
        validation_value = val_values[idx] if idx < len(val_values) else None
        rows.append(
            {
                "round": idx,
                "train_value": float(train_value),
                "validation_value": None if validation_value is None else float(validation_value),
            }
        )
    return metric_name, rows


def load_retraining_snapshots(path: Path = DEFAULT_RETRAINING_SNAPSHOTS_PATH) -> list[dict[str, Any]]:
    """Load retraining snapshot metrics."""
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, Any]] = []
        for idx, row in enumerate(reader):
            rows.append(
                {
                    "snapshot_index": idx + 1,
                    "timestamp_utc": str(row.get("timestamp_utc") or ""),
                    "clip_count": int(float(row.get("clip_count") or 0)),
                    "row_count": int(float(row.get("row_count") or 0)),
                    "accuracy": float(row.get("accuracy") or 0.0),
                    "f1": float(row.get("f1") or 0.0),
                    "auc": None if not str(row.get("auc") or "").strip() else float(row["auc"]),
                    "best_iteration": int(float(row.get("best_iteration") or 0)),
                }
            )
    return rows


def load_feature_importance(path: Path = DEFAULT_XGB_MODEL_PATH, top_n: int = 12) -> list[dict[str, float | str]]:
    """Load top transformed feature importances from the trained model bundle."""
    try:
        import joblib  # type: ignore[import-not-found]
    except Exception:
        return []

    if not path.exists():
        return []

    bundle = joblib.load(path)
    if not isinstance(bundle, dict):
        return []

    preprocessor = bundle.get("preprocessor")
    classifier = bundle.get("classifier")
    if preprocessor is None or classifier is None:
        return []

    importances = getattr(classifier, "feature_importances_", None)
    if importances is None:
        return []

    feature_names = list(preprocessor.get_feature_names_out())
    pairs = sorted(
        (
            {"feature": str(name), "importance": float(score)}
            for name, score in zip(feature_names, importances, strict=False)
        ),
        key=lambda item: item["importance"],
        reverse=True,
    )
    return pairs[:top_n]


def load_learning_loop_report() -> dict[str, Any]:
    """Aggregate persisted learning-loop data for the UI."""
    metric_name, training_history = load_training_history()
    retraining_snapshots = load_retraining_snapshots()
    feature_importance = load_feature_importance()
    latest_snapshot = retraining_snapshots[-1] if retraining_snapshots else None

    return {
        "metric_name": metric_name,
        "training_history": training_history,
        "retraining_snapshots": retraining_snapshots,
        "feature_importance": feature_importance,
        "summary": {
            "history_rounds": len(training_history),
            "snapshot_count": len(retraining_snapshots),
            "latest_clip_count": latest_snapshot["clip_count"] if latest_snapshot else None,
            "latest_row_count": latest_snapshot["row_count"] if latest_snapshot else None,
            "latest_accuracy": latest_snapshot["accuracy"] if latest_snapshot else None,
            "latest_f1": latest_snapshot["f1"] if latest_snapshot else None,
            "latest_auc": latest_snapshot["auc"] if latest_snapshot else None,
        },
    }
