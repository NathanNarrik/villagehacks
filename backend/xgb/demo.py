"""Small end-to-end demo for the word-risk model."""

from __future__ import annotations

import argparse
from pathlib import Path

from .features import DEFAULT_DATASET_PATH, ensure_directories, load_scribe_payloads, scribe_words_from_payload
from .infer import score_transcript_words
from .learning_loop import load_state
from .train import train_model
from .viz import plot_feature_importance, plot_retraining_snapshots, plot_training_history


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ScribeShield XGBoost demo.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--corrected", type=Path, default=None)
    parser.add_argument("--scribe", type=Path, default=None)
    parser.add_argument("--sample-clip-id", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    ensure_directories()

    if args.rebuild:
        missing = [str(path) for path in (args.manifest, args.corrected, args.scribe) if path is None or not path.exists()]
        if missing:
            raise SystemExit("Missing required input path(s): " + ", ".join(missing))

    result = train_model(
        dataset_path=args.dataset,
        rebuild=bool(args.rebuild),
        manifest_path=args.manifest,
        corrected_path=args.corrected,
        scribe_path=args.scribe,
    )

    print(f"Accuracy: {result.accuracy:.4f}")
    print(f"F1: {result.f1:.4f}")
    if result.auc is not None:
        print(f"AUC: {result.auc:.4f}")

    print(f"Chart: {plot_training_history()}")
    print(f"Chart: {plot_retraining_snapshots()}")
    print(f"Chart: {plot_feature_importance()}")

    if args.scribe is None:
        print("No sample transcript scored because --scribe was not provided.")
        return

    payloads = load_scribe_payloads(args.scribe)
    if not payloads:
        raise SystemExit(f"No Scribe JSON inputs found at {args.scribe}")

    clip_id = args.sample_clip_id or next(iter(payloads))
    payload = payloads.get(clip_id)
    if payload is None:
        raise SystemExit(f"Sample clip_id {clip_id!r} was not found in {args.scribe}")

    learning_state = load_state()
    inference = score_transcript_words(
        clip_id=clip_id,
        words=scribe_words_from_payload(payload),
        clip_metadata={},
        keyterms=set(learning_state.keyterm_counts),
        correction_frequency=learning_state.correction_frequency,
    )
    if inference is None:
        print("Model bundle unavailable for sample scoring.")
        return

    top_scores = sorted(inference.word_scores, key=lambda item: item.risk, reverse=True)[:10]
    print("Top 10 risky words:")
    for item in top_scores:
        print(f"- {item.word} [{item.start_ms}-{item.end_ms}] risk={item.risk:.3f}")


if __name__ == "__main__":
    main()
