from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_run_benchmark_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_benchmark.py"
    spec = importlib.util.spec_from_file_location("run_benchmark", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _baseline_payload() -> dict:
    return {
        "results": [
            {
                "clip_id": "clip_a",
                "category": "Medication Refill",
                "difficulty": "Standard",
                "raw_wer": 0.2,
                "corrected_wer": 0.1,
                "improvement_pct": 50.0,
            }
        ],
        "ablation": [],
        "metrics": {
            "verification_rate": 0.9,
            "unsafe_guess_rate": 0.0,
            "uncertainty_coverage": 0.8,
            "phonetic_hit_rate": 0.7,
        },
        "aggregate": {
            "avg_raw_wer": 0.2,
            "avg_corrected_wer": 0.1,
            "avg_improvement_pct": 50.0,
            "keyterm_impact_pct": 12.0,
        },
    }


def test_recompute_fallback_adds_optional_fields(tmp_path):
    mod = _load_run_benchmark_module()
    payload = _baseline_payload()

    out = mod._recompute(payload=payload, eval_path=tmp_path / "missing.jsonl")

    row = out["results"][0]
    assert row["raw_cer"] is None
    assert row["corrected_cer"] is None
    assert row["raw_digit_accuracy"] is None
    assert row["corrected_digit_accuracy"] is None
    assert row["raw_medical_keyword_accuracy"] is None
    assert row["corrected_medical_keyword_accuracy"] is None

    agg = out["aggregate"]
    assert agg["avg_raw_cer"] is None
    assert agg["avg_corrected_cer"] is None
    assert agg["avg_raw_digit_accuracy"] is None
    assert agg["avg_corrected_digit_accuracy"] is None
    assert agg["avg_raw_medical_keyword_accuracy"] is None
    assert agg["avg_corrected_medical_keyword_accuracy"] is None


def test_recompute_metrics_from_eval_rows(tmp_path):
    mod = _load_run_benchmark_module()
    payload = _baseline_payload()

    eval_row = {
        "clip_id": "clip_a",
        "category": "Medication Refill",
        "difficulty": "Standard",
        "ground_truth": "I take metformin 50 mg once daily",
        "raw_text": "I take metfornin 15 mg once daily",
        "corrected_text": "I take metformin 50 mg once daily",
    }

    eval_path = tmp_path / "benchmark_eval.jsonl"
    eval_path.write_text(json.dumps(eval_row) + "\n", encoding="utf-8")

    out = mod._recompute(payload=payload, eval_path=eval_path)

    row = out["results"][0]
    assert row["clip_id"] == "clip_a"
    assert row["corrected_wer"] <= row["raw_wer"]
    assert row["corrected_cer"] <= row["raw_cer"]
    assert row["corrected_digit_accuracy"] >= row["raw_digit_accuracy"]
    assert row["corrected_medical_keyword_accuracy"] >= row["raw_medical_keyword_accuracy"]

    agg = out["aggregate"]
    assert agg["avg_raw_cer"] is not None
    assert agg["avg_corrected_cer"] is not None
    assert agg["avg_raw_digit_accuracy"] is not None
    assert agg["avg_corrected_digit_accuracy"] is not None
    assert agg["avg_raw_medical_keyword_accuracy"] is not None
    assert agg["avg_corrected_medical_keyword_accuracy"] is not None
