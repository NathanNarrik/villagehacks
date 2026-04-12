"""Generate benchmark_results.json for the demo backend.

This script currently writes a reproducible baseline artifact derived from the
project's committed benchmark template. It provides a stable fallback contract
for `/benchmark` and can be replaced with a full WER recomputation pipeline.
"""
from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    backend_dir = Path(__file__).resolve().parents[1]
    src = backend_dir / "data" / "benchmark_results.json"
    if not src.exists():
        src = backend_dir / "data" / "benchmark_results.json.example"
    dst = backend_dir / "data" / "benchmark_results.json"

    if not src.exists():
        raise SystemExit("No benchmark source artifact found.")

    payload = json.loads(src.read_text(encoding="utf-8"))
    dst.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote benchmark artifact: {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
