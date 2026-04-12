# Test Audio Layout

This directory keeps demo audio and benchmark audio separate so we can ship polished demos without breaking benchmark comparability.

## Structure

```text
backend/test_audio/
  demo/
    manifest.csv
    audio/
  benchmark/
    v1/
      manifest.csv
      audio/
        standard/
        adversarial/
```

## Rules

1. Benchmark data is versioned and stable.
2. Do not rename benchmark `clip_id`s once a version is in use.
3. If benchmark content changes, create `benchmark/v2/` instead of modifying `benchmark/v1/`.
4. Demo audio can be refreshed any time; keep a row in `demo/manifest.csv` for each clip.

## Naming Conventions

- Benchmark audio: `clip_01.wav` ... `clip_20.wav` (matches `clip_id` exactly)
- Demo audio: `demo_YYYYMMDD_<scenario>_takeNN.wav`

## Metadata Source of Truth

- Benchmark mapping: `benchmark/v1/manifest.csv`
- Demo mapping: `demo/manifest.csv`

Use relative paths in manifests, rooted at each manifest's directory.
