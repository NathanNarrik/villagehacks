"""Training utilities for the word-risk XGBoost model."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .features import (
    CATEGORICAL_FEATURES,
    DEFAULT_DATASET_PATH,
    DEFAULT_FEATURE_SCHEMA_PATH,
    DEFAULT_RETRAINING_SNAPSHOTS_PATH,
    DEFAULT_TRAINING_HISTORY_PATH,
    DEFAULT_XGB_MODEL_PATH,
    NUMERIC_FEATURES,
    build_training_rows,
    ensure_directories,
    load_corrected_rows,
    load_manifest_rows,
    load_scribe_payloads,
    write_dataset_rows,
)
from .learning_loop import load_state


@dataclass(slots=True)
class TrainingResult:
    """Result summary for one training run."""

    accuracy: float
    f1: float
    auc: float | None
    best_iteration: int
    row_count: int
    clip_count: int
    artifact_path: Path


def _require_ml_dependencies() -> tuple[Any, ...]:
    try:
        import joblib  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
        import pandas as pd  # type: ignore[import-not-found]
        from sklearn.compose import ColumnTransformer  # type: ignore[import-not-found]
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score  # type: ignore[import-not-found]
        from sklearn.model_selection import train_test_split  # type: ignore[import-not-found]
        from sklearn.preprocessing import OneHotEncoder  # type: ignore[import-not-found]
        from xgboost import XGBClassifier  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Training requires pandas, numpy, scikit-learn, joblib, and xgboost."
        ) from exc
    return (
        joblib,
        np,
        pd,
        ColumnTransformer,
        accuracy_score,
        f1_score,
        roc_auc_score,
        train_test_split,
        OneHotEncoder,
        XGBClassifier,
    )


def _snapshot_fieldnames() -> list[str]:
    return [
        "timestamp_utc",
        "clip_count",
        "row_count",
        "accuracy",
        "f1",
        "auc",
        "best_iteration",
    ]


def _append_snapshot(
    *,
    result: TrainingResult,
    path: Path = DEFAULT_RETRAINING_SNAPSHOTS_PATH,
) -> None:
    ensure_directories()
    file_exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_snapshot_fieldnames())
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "clip_count": result.clip_count,
                "row_count": result.row_count,
                "accuracy": f"{result.accuracy:.6f}",
                "f1": f"{result.f1:.6f}",
                "auc": "" if result.auc is None else f"{result.auc:.6f}",
                "best_iteration": result.best_iteration,
            }
        )


def build_dataset_from_inputs(
    *,
    manifest_path: Path,
    corrected_path: Path,
    scribe_path: Path,
    dataset_path: Path = DEFAULT_DATASET_PATH,
) -> Path:
    """Build and persist the training dataset from source inputs."""
    manifest_rows = load_manifest_rows(manifest_path)
    corrected_rows = load_corrected_rows(corrected_path)
    scribe_payloads = load_scribe_payloads(scribe_path)
    learning_state = load_state()
    rows = build_training_rows(
        manifest_rows=manifest_rows,
        corrected_rows=corrected_rows,
        scribe_payloads=scribe_payloads,
        keyterms=set(learning_state.keyterm_counts) or None,
        correction_frequency=learning_state.correction_frequency,
    )
    write_dataset_rows(rows, dataset_path)
    return dataset_path


def _load_or_build_dataset(
    *,
    dataset_path: Path,
    rebuild: bool,
    manifest_path: Path | None,
    corrected_path: Path | None,
    scribe_path: Path | None,
) -> tuple[Any, Path]:
    (
        _joblib,
        _np,
        pd,
        _ColumnTransformer,
        _accuracy_score,
        _f1_score,
        _roc_auc_score,
        _train_test_split,
        _OneHotEncoder,
        _XGBClassifier,
    ) = _require_ml_dependencies()

    if rebuild:
        missing = [
            str(path)
            for path in (manifest_path, corrected_path, scribe_path)
            if path is None or not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                "Cannot rebuild dataset; missing source paths: " + ", ".join(missing)
            )
        assert manifest_path is not None
        assert corrected_path is not None
        assert scribe_path is not None
        dataset_path = build_dataset_from_inputs(
            manifest_path=manifest_path,
            corrected_path=corrected_path,
            scribe_path=scribe_path,
            dataset_path=dataset_path,
        )

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. "
            "Provide --rebuild with manifest/corrected/scribe paths to generate it."
        )

    frame = pd.read_csv(dataset_path)
    if "needs_verification" not in frame.columns:
        raise ValueError("Dataset is missing the needs_verification label column.")
    return frame, dataset_path


def _safe_stratify(series: Any) -> Any | None:
    counts = series.value_counts()
    return series if len(counts) > 1 and int(counts.min()) >= 2 else None


def _split_frame(frame: Any, train_test_split: Any) -> tuple[Any, Any, Any]:
    if "split" in frame.columns and frame["split"].astype(str).str.len().any():
        train = frame[frame["split"] == "train"].copy()
        val = frame[frame["split"] == "val"].copy()
        test = frame[frame["split"] == "test"].copy()
        if val.empty and not test.empty:
            val = test.copy()
        if train.empty:
            raise ValueError("Dataset split column is present but contains no train rows.")
        return train, val, test

    labeled = frame.copy()
    stratify = _safe_stratify(labeled["needs_verification"])
    train, test = train_test_split(
        labeled,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )
    stratify_train = _safe_stratify(train["needs_verification"])
    train, val = train_test_split(
        train,
        test_size=0.25,
        random_state=42,
        stratify=stratify_train,
    )
    return train.copy(), val.copy(), test.copy()


def train_model(
    *,
    dataset_path: Path = DEFAULT_DATASET_PATH,
    rebuild: bool = False,
    manifest_path: Path | None = None,
    corrected_path: Path | None = None,
    scribe_path: Path | None = None,
    artifact_path: Path = DEFAULT_XGB_MODEL_PATH,
    feature_schema_path: Path = DEFAULT_FEATURE_SCHEMA_PATH,
    history_path: Path = DEFAULT_TRAINING_HISTORY_PATH,
    snapshots_path: Path = DEFAULT_RETRAINING_SNAPSHOTS_PATH,
    low_threshold: float = 0.6,
    medium_threshold: float = 0.35,
) -> TrainingResult:
    """Train the binary word-risk model and persist artifacts."""
    (
        joblib,
        _np,
        pd,
        ColumnTransformer,
        accuracy_score,
        f1_score,
        roc_auc_score,
        train_test_split,
        OneHotEncoder,
        XGBClassifier,
    ) = _require_ml_dependencies()

    frame, dataset_path = _load_or_build_dataset(
        dataset_path=dataset_path,
        rebuild=rebuild,
        manifest_path=manifest_path,
        corrected_path=corrected_path,
        scribe_path=scribe_path,
    )
    train_frame, val_frame, test_frame = _split_frame(frame, train_test_split)

    raw_feature_names = CATEGORICAL_FEATURES + NUMERIC_FEATURES
    for name in raw_feature_names:
        if name not in frame.columns:
            raise ValueError(f"Dataset is missing feature column {name!r}")

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ],
        remainder="passthrough",
    )

    x_train = train_frame[raw_feature_names]
    y_train = train_frame["needs_verification"].astype(int)
    x_val = val_frame[raw_feature_names] if not val_frame.empty else train_frame[raw_feature_names]
    y_val = val_frame["needs_verification"].astype(int) if not val_frame.empty else y_train

    x_train_encoded = preprocessor.fit_transform(x_train)
    x_val_encoded = preprocessor.transform(x_val)

    classifier = XGBClassifier(
        objective="binary:logistic",
        eval_metric=["logloss"],
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        random_state=42,
        early_stopping_rounds=20,
    )
    classifier.fit(
        x_train_encoded,
        y_train,
        eval_set=[(x_train_encoded, y_train), (x_val_encoded, y_val)],
        verbose=False,
    )

    evaluation_frame = test_frame if not test_frame.empty else val_frame
    x_eval = evaluation_frame[raw_feature_names]
    y_eval = evaluation_frame["needs_verification"].astype(int)
    x_eval_encoded = preprocessor.transform(x_eval)
    y_pred = classifier.predict(x_eval_encoded)
    y_prob = classifier.predict_proba(x_eval_encoded)[:, 1]

    accuracy = float(accuracy_score(y_eval, y_pred))
    f1 = float(f1_score(y_eval, y_pred, zero_division=0))
    auc: float | None = None
    if len(set(int(value) for value in y_eval.tolist())) > 1:
        auc = float(roc_auc_score(y_eval, y_prob))

    ensure_directories()
    bundle = {
        "preprocessor": preprocessor,
        "classifier": classifier,
        "raw_feature_names": raw_feature_names,
        "categorical_feature_names": CATEGORICAL_FEATURES,
        "numeric_feature_names": NUMERIC_FEATURES,
        "label_name": "needs_verification",
        "thresholds": {
            "low": low_threshold,
            "medium": medium_threshold,
        },
        "dataset_path": str(dataset_path),
    }
    joblib.dump(bundle, artifact_path)

    history = classifier.evals_result()
    history_path.write_text(json.dumps(history, indent=2, sort_keys=True), encoding="utf-8")

    feature_schema_payload = {
        "raw_feature_names": raw_feature_names,
        "categorical_feature_names": CATEGORICAL_FEATURES,
        "numeric_feature_names": NUMERIC_FEATURES,
        "label_name": "needs_verification",
        "dataset_columns": raw_feature_names + ["needs_verification"],
    }
    feature_schema_path.write_text(
        json.dumps(feature_schema_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    result = TrainingResult(
        accuracy=accuracy,
        f1=f1,
        auc=auc,
        best_iteration=int(getattr(classifier, "best_iteration", -1)),
        row_count=int(len(frame)),
        clip_count=int(frame["clip_id"].nunique()) if "clip_id" in frame.columns else int(len(frame)),
        artifact_path=artifact_path,
    )
    _append_snapshot(result=result, path=snapshots_path)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the ScribeShield word-risk XGBoost model.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--corrected", type=Path, default=None)
    parser.add_argument("--scribe", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = train_model(
        dataset_path=args.dataset,
        rebuild=bool(args.rebuild),
        manifest_path=args.manifest,
        corrected_path=args.corrected,
        scribe_path=args.scribe,
    )
    print(f"artifact={result.artifact_path}")
    print(f"accuracy={result.accuracy:.4f}")
    print(f"f1={result.f1:.4f}")
    if result.auc is not None:
        print(f"auc={result.auc:.4f}")


if __name__ == "__main__":
    main()
