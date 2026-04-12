"""Visualization helpers for the word-risk model."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .features import (
    DEFAULT_RETRAINING_SNAPSHOTS_PATH,
    DEFAULT_TRAINING_HISTORY_PATH,
    DEFAULT_XGB_MODEL_PATH,
    REPORTS_DIR,
    ensure_directories,
)


def _require_plot_dependencies() -> tuple[Any, Any]:
    try:
        import joblib  # type: ignore[import-not-found]
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Visualization requires matplotlib and joblib.") from exc
    return joblib, plt


def plot_training_history(
    *,
    history_path: Path = DEFAULT_TRAINING_HISTORY_PATH,
    output_path: Path | None = None,
) -> Path:
    """Plot train and validation metrics over boosting rounds."""
    _joblib, plt = _require_plot_dependencies()
    ensure_directories()
    output = output_path or REPORTS_DIR / "training_history.png"
    payload = json.loads(history_path.read_text(encoding="utf-8"))

    train_metrics = payload.get("validation_0") or {}
    val_metrics = payload.get("validation_1") or {}
    metric_name = next(iter(train_metrics.keys()), "logloss")
    train_values = list(train_metrics.get(metric_name) or [])
    val_values = list(val_metrics.get(metric_name) or [])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(range(len(train_values)), train_values, label="train")
    if val_values:
        ax.plot(range(len(val_values)), val_values, label="validation")
    ax.set_title(f"Training vs Validation {metric_name}")
    ax.set_xlabel("Boosting round")
    ax.set_ylabel(metric_name)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return output


def plot_retraining_snapshots(
    *,
    snapshots_path: Path = DEFAULT_RETRAINING_SNAPSHOTS_PATH,
    output_path: Path | None = None,
) -> Path:
    """Plot improvement over retraining snapshots."""
    _joblib, plt = _require_plot_dependencies()
    ensure_directories()
    output = output_path or REPORTS_DIR / "retraining_snapshots.png"
    rows: list[dict[str, str]] = []
    with snapshots_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]

    x_values = [int(row["clip_count"] or 0) for row in rows]
    accuracy_values = [float(row["accuracy"] or 0.0) for row in rows]
    f1_values = [float(row["f1"] or 0.0) for row in rows]
    auc_values = [
        float(row["auc"])
        for row in rows
        if str(row.get("auc") or "").strip()
    ]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x_values, accuracy_values, marker="o", label="accuracy")
    ax.plot(x_values, f1_values, marker="o", label="f1")
    if auc_values and len(auc_values) == len(x_values):
        ax.plot(x_values, auc_values, marker="o", label="auc")
    ax.set_title("Model Improvement Over Retraining")
    ax.set_xlabel("Cumulative corrected calls")
    ax.set_ylabel("Metric")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return output


def plot_feature_importance(
    *,
    model_path: Path = DEFAULT_XGB_MODEL_PATH,
    output_path: Path | None = None,
    top_n: int = 20,
) -> Path:
    """Plot top feature importances from the trained bundle."""
    joblib, plt = _require_plot_dependencies()
    ensure_directories()
    output = output_path or REPORTS_DIR / "feature_importance.png"
    bundle = joblib.load(model_path)
    preprocessor = bundle["preprocessor"]
    classifier = bundle["classifier"]

    feature_names = list(preprocessor.get_feature_names_out())
    importances = list(getattr(classifier, "feature_importances_", []))
    pairs = sorted(zip(feature_names, importances), key=lambda item: item[1], reverse=True)[:top_n]
    labels = [name for name, _ in pairs]
    values = [float(value) for _, value in pairs]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(labels[::-1], values[::-1])
    ax.set_title("Top Feature Importances")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return output
