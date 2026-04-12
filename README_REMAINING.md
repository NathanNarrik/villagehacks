# CareCaller AI - Remaining Implementation Guide

This document is an engineering handoff for the work that is still left to finish in this repo. It is based on the April 2026 final plan, but it is grounded in the code and assets that currently exist in the workspace.

The goal of this README is not to restate the pitch. It is to help the next contributor understand:

- what is already implemented
- what is only partially implemented
- what still needs to be built
- what is intentionally out of scope for this repo version

## Status Snapshot

- Core batch pipeline exists end to end in the backend: preprocessing -> Scribe batch transcription -> uncertainty scoring -> Tavily verification -> Claude correction -> Claude extraction -> learning loop.
- Landing, demo, and benchmark pages exist in the frontend.
- XGBoost is partially integrated as a post-Scribe risk scorer, not as a replacement for Scribe.
- Redis persistence and full benchmark automation parity are still incomplete.
- The supported product flow for this repo version is **demo audio clips only**.
- The final demo audio set is now generated from the canonical six-row demo CSV and exported into both backend and frontend asset folders.

## Update (April 12, 2026)

Completed in this repo iteration:

- In-memory storage is now explicitly formalized as the active runtime mode (no Redis dependency added).
- `/health` reports the active store mode from the storage layer.
- Benchmark manifest metadata (`ground_truth`, `medical_keywords`) is now populated in
  `backend/test_audio/benchmark/v1/manifest.csv`.
- `backend/scripts/run_benchmark.py` now reads benchmark manifest metadata and applies it during recomputation.
- Frontend benchmark rendering now normalizes ratio-vs-percent API data and uses real ablation-driven trend data instead of synthetic random trend values.
- Frontend API types were aligned with backend schemas for benchmark optional metrics and health payload fields.
- Frontend stale placeholder page was reconciled (`frontend/src/pages/Index.tsx` now redirects cleanly).

## Scope Decisions

- `POST /transcribe` is the primary supported demo API.
- The frontend should stay centered on fixed demo clips, not `Upload` or `Record`.
- `Upload` and `Record` UI flows from the final plan are intentionally **out of scope** for this repo version.
- `/stream` and `/stream/token` may stay in the backend as capability or future work, but they are not required to complete the current demo flow.
- The current root [README.md](/Users/aayanmapara/Hackathons/villagehacks/README.md) should remain the main setup/run guide. This file is the remaining-work handoff.

## Current State vs Final Plan

### Already Implemented

- Backend batch pipeline orchestration exists in `backend/app/pipeline.py`.
- Audio preprocessing exists and is wired through the backend in `backend/app/preprocessing.py` and `backend/audio_preprocess/`.
- ElevenLabs Scribe batch transcription exists in `backend/app/scribe.py`.
- Rule-based multi-signal uncertainty scoring exists in `backend/app/uncertainty.py`.
- Tavily verification exists in `backend/app/tavily_verify.py`.
- Claude safe correction with a hallucination guard exists in `backend/app/claude_correct.py`.
- Claude extraction for structured clinical output exists in `backend/app/claude_extract.py`.
- Learning-loop state and adaptive keyterm logic exist in `backend/app/learning_loop.py`.
- The FastAPI surface already includes `/transcribe`, `/benchmark`, `/health`, `/stream/token`, and `/stream`.
- Frontend routing already includes landing, demo, and benchmark pages in `frontend/src/App.tsx`.
- The demo page already runs fixed clip files through the live backend via `POST /transcribe`.
- The benchmark page already renders benchmark results and can read from the backend API.
- XGBoost training assets already exist:
  - `backend/audio_gen/train_xgboost_telephony.py`
  - `backend/audio_gen/README_XGBOOST_TELEPHONY.md`
  - `backend/audio_gen/xgboost_telephony_model.joblib`
- Runtime XGBoost loading is already wired in `backend/app/uncertainty.py`.

### Partially Implemented

#### Backend / Pipeline

- The final-plan 7-layer pipeline mostly exists, but some layers are simplified relative to the original spec.
- Speaker resolution is heuristic and simple, not deeply productized.
- The learning loop works in process, but not with durable shared persistence.
- Realtime endpoints exist, but they are not part of the supported demo scope for this repo version.

#### Persistence / Infra

- Storage is still in-memory via `backend/app/storage.py`.
- The final plan called for Redis-backed persistence across calls and restarts. That is not done yet.
- Health output currently reports `redis: "in-memory"` rather than a real Redis connection.

#### Benchmarking

- `/benchmark` currently serves `backend/data/benchmark_results.json`.
- `backend/scripts/run_benchmark.py` can regenerate or recompute benchmark output, but the repo is still closer to a cached-results workflow than the full final-plan benchmark service.
- Benchmark manifests exist under `backend/test_audio/benchmark/v1/`, but key metadata fields like ground truth and medical keywords are still blank in `manifest.csv`.
- The benchmark page still contains some presentation-layer simplifications, including mocked learning-loop trend data instead of a fully real trend derived from benchmark runs.

#### Frontend

- The app already supports the intended fixed-demo-clip flow.
- The frontend is optimized around hardcoded scenarios in `frontend/src/pages/DemoPage.tsx`, not a manifest-driven clip system yet.
- An `/about` route from the final plan is missing.
- `frontend/src/pages/Index.tsx` is still a stale placeholder page and should either be removed or reconciled.
- The current UI is useful for the demo, but it does not yet represent full final-plan parity across every planned interaction.

#### ML / XGBoost

- XGBoost is already wired as a post-Scribe risk signal.
- Current runtime usage is lighter than the full intended architecture:
  - the model loads
  - the scorer runs
  - `xgboost_risk:*` can be appended to uncertainty signals
- The richer transcript-derived feature path from the final plan is still incomplete.

### Still To Build

- Durable Redis-backed learning-loop and cache storage.
- Full benchmark asset population and tighter automation around `backend/test_audio/benchmark/v1`.
- A stronger XGBoost feature builder based on Scribe transcript output.
- Final deployment documentation and deployment parity with the original architecture.

## Demo Audio Generation

This workflow is now implemented.

The repo now has four coordinated pieces in the demo-audio story:

- `backend/audio_gen/input/demo_cards_20260412.csv`
  - canonical synthesis spec for the six shipped situations
- `backend/test_audio/demo/scripts/*.txt`
  - source scripts that must stay text-identical to the canonical CSV
- `backend/test_audio/demo/manifest.csv`
  - checked-in mapping for backend demo WAVs and frontend public WAVs
- `frontend/public/demo-audio/<situation>/*.wav`
  - the shipped public per-take demo assets used by the frontend demo page
- `frontend/public/demo-audio/*.wav`
  - top-level compatibility aliases for a few signature takes

### Current Reality

- The frontend demo now works by selecting one of six situations, then one of four takes for that situation.
- `conda run -n village-hacks python -m backend.audio_gen.build_demo_audio` is the repeatable command for regenerating the shipped demo set.
- The wrapper validates the canonical CSV against `backend/test_audio/demo/scripts/*.txt`, expands the six situations into 24 takes, generates telephony audio under `backend/audio_gen/output/demo_cards_20260412/`, and exports the shipped WAVs into both backend and frontend asset folders.
- Ambient/noisy variants are remixed with richer background beds like conversation, TV, music, and room tone rather than static-only noise.
- `backend/test_audio/demo/manifest.csv` now records the canonical clip ids, backend demo WAV paths, script references, and frontend public WAV paths for all 24 takes.

### Current Demo-Audio Contract

- Canonical generation spec: `backend/audio_gen/input/demo_cards_20260412.csv`
- Canonical script text: `backend/test_audio/demo/scripts/*.txt`
- Exported backend demo WAVs: `backend/test_audio/demo/audio/`
- Exported frontend demo WAVs: `frontend/public/demo-audio/<situation>/`
- Top-level frontend aliases: `frontend/public/demo-audio/*.wav`
- Checked-in mapping: `backend/test_audio/demo/manifest.csv`
- Frontend situation metadata remains hardcoded in `frontend/src/pages/DemoPage.tsx`, but the situation ids now match the canonical clip ids from the demo CSV.

## Remaining Workstreams

### 1. Environment And Tests

- Align the conda environment naming and setup docs.
  - `backend/environment.yml` currently declares `villagehacks-audio`.
  - The locally available env discovered during validation was `village-hacks`.
- Make sure the chosen backend conda env actually contains application and test dependencies.
  - A local validation run with `conda run -n village-hacks python -m pytest -q` failed because packages like `anthropic` were missing.
- Update docs so the intended backend test flow is reliable and reproducible.

### 2. Demo Audio Pipeline

- Re-run `conda run -n village-hacks python -m backend.audio_gen.build_demo_audio` whenever the six demo scripts or canonical demo CSV change.
- Keep the exported backend/frontend WAVs committed and aligned with the manifest after each regeneration.
- Preserve the current nested frontend public structure and the top-level compatibility aliases because the live demo page depends on them.

### 3. Benchmark Pipeline

- Populate `backend/test_audio/benchmark/v1/manifest.csv` with real:
  - ground truth
  - medical keywords
  - any additional benchmark metadata needed for recomputation
- Tie benchmark recomputation more directly to `backend/test_audio/benchmark/v1` instead of leaning mainly on cached JSON.
- Decide whether benchmark outputs should be committed demo artifacts, generated outputs, or both.
- Replace any synthetic frontend-only benchmark storytelling with real benchmark-derived data where practical.

### 4. Persistence And Deployment

- Replace in-memory storage in `backend/app/storage.py` with Redis, or clearly formalize in-memory mode as hackathon-only fallback.
- Persist:
  - phonetic map
  - learned keyterms
  - Tavily cache
  - any future shared runtime state
- Document and eventually implement the intended deployment split:
  - frontend on Vercel
  - backend plus persistence on Railway or AWS

### 5. Frontend Polish Still Relevant To Demo-Clip Scope

- Keep the frontend centered on fixed demo clips only.
- Remove or reconcile stale placeholder code such as `frontend/src/pages/Index.tsx`.
- Decide whether `/about` is still worth adding for the final demo package.
- Keep benchmark and demo UX improvements focused on the shipped clip flow, not on upload/record flows that are out of scope.

### 6. XGBoost Integration

The intended architecture is:

`Scribe transcript -> feature builder -> XGBoost risk score -> Tavily/Claude if risky`

That means:

- Scribe still performs transcription.
- XGBoost does **not** replace Scribe.
- XGBoost acts as a lightweight post-processing risk layer over the Scribe output.

#### What Already Exists

- Training script
- Saved model artifact
- Config path for the saved model
- Runtime scorer hook in the uncertainty layer

#### What Is Still Missing

- A richer word-level or span-level feature builder using Scribe output such as:
  - token text
  - previous and next token context
  - speaker label
  - token duration
  - pauses before and after
  - numeric flags
  - medical-term candidate flags
  - keyterm matches
  - phonetic distance to nearest keyterm
  - timing irregularity
  - any reliable uncertainty proxy from Scribe output
- A clear training label schema, such as:
  - `needs_verification`
  - `needs_correction`
- Threshold calibration for LOW and MEDIUM risk handling.
- Offline evaluation showing whether XGBoost improves verification gating over the rule-based path.
- Clearer runtime integration so XGBoost is a real decision layer, not just a weak add-on signal.

## Public Interfaces To Keep In Mind

These are the interfaces that should stay stable or be changed carefully.

### Primary Supported Endpoint

- `POST /transcribe`
  - This is the main supported endpoint for the current demo flow.
  - The frontend demo page depends on it.

### Benchmark Endpoint

- `GET /benchmark`
  - Current behavior is based on serving or regenerating benchmark JSON.
  - This differs from the more ambitious final-plan description of a fully automated benchmark service.
  - The README and future work should document that difference honestly.

### Health Endpoint

- `GET /health`
  - Useful for surfacing learning-loop counts and runtime dependency status.
  - It should stay honest about in-memory versus Redis-backed operation.

### Frontend Types

- Frontend API types in `frontend/src/types/api.ts` should be reviewed against the actual backend response models in `backend/app/schemas.py`.
- There are already some signs of drift:
  - the backend health model includes fields the frontend type does not fully capture
  - benchmark outputs include richer fields than the frontend always uses

## Out Of Scope For This Repo Version

The following items were in the final plan, but they are intentionally not part of the remaining implementation target for this repo version:

- Upload-based demo flow
- Record-from-mic demo flow
- Live waveform and browser recording UI
- Upload/record tab system
- Building the frontend primarily around realtime streaming

Backend realtime support may remain in the codebase, but it is not required to finish the fixed-demo-clip experience.

## Test And Verification Plan

### Backend

1. Activate the intended conda env.
2. Install backend dependencies from `backend/requirements.txt` or the finalized environment definition.
3. Run:

```bash
cd backend
python -m pytest -q
```

### Frontend

1. Run:

```bash
cd frontend
npm test
```

2. Verify the landing, demo, and benchmark pages render correctly.

### Demo Clips

- Confirm every clip referenced by the demo page exists in `frontend/public/demo-audio/`.
- Confirm each clip successfully calls `POST /transcribe`.
- Confirm the demo page renders:
  - raw transcript
  - corrected transcript
  - clinical summary
  - latency values

### Benchmark

1. Regenerate benchmark results:

```bash
cd backend
python scripts/run_benchmark.py
```

2. Confirm `/benchmark` serves the expected JSON shape.
3. Confirm the benchmark page renders current API data without falling back unintentionally to mock data.

### XGBoost

- Confirm the model file loads from `backend/audio_gen/xgboost_telephony_model.joblib`.
- Confirm uncertainty signals can include `xgboost_risk:*` when the scorer is active.
- Confirm the system falls back cleanly to rule-based uncertainty scoring when the model or ML dependencies are unavailable.

## Assumptions And Defaults

- This file is the remaining-work guide. The root [README.md](/Users/aayanmapara/Hackathons/villagehacks/README.md) remains the main setup/run doc.
- Demo flow is **fixed demo audios only**.
- Generating the final demo audios is still an explicit required task and should stay near the top of the remaining-work discussion.
- XGBoost should be presented as a **partial integration already present in code**, not as a future-only idea.
- The right next milestone is not "build every page from the original plan." It is "finish the demo-audio-centered product path cleanly and honestly."
