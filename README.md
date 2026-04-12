# CareCaller AI — Backend

FastAPI backend for the CareCaller AI verification-augmented STT pipeline.

## Quick start

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS/Linux
pip install -r requirements.txt
copy .env.example .env             # then fill in API keys
uvicorn app.main:app --reload --port 8000
```

The frontend should point at `http://localhost:8000` (set `VITE_API_URL` if needed).

## What you need to fill in

1. **`backend/.env`** — paste these three keys:
   - `ANTHROPIC_API_KEY` — for Claude correction + extraction (Person B)
   - `TAVILY_API_KEY` — for medical-term verification (Person B)
   - `ELEVENLABS_API_KEY` — for Scribe v2 transcription (Person A's module)
2. Ensure `backend/data/benchmark_results.json` exists (included in this repo).

## Endpoints

| Method | Path | Notes |
|---|---|---|
| POST | `/transcribe` | multipart `file=<audio>`. Runs full batch pipeline. |
| GET  | `/stream/token` | Returns single-use token for websocket auth (`expires_in=60`). |
| WS   | `/stream?token=...` | Realtime Scribe relay. Emits `partial`, `committed`, `correction`, `error` events. |
| GET  | `/benchmark?clips=all\|adversarial\|standard` | Serves benchmark JSON from `data/benchmark_results.json`. |
| GET  | `/health` | Reachability + runtime status (including learning loop stats + realtime dependency status). |

## Architecture (7 layers)

1. **preprocessing** — ffmpeg loudnorm + denoise
2. **scribe** — ElevenLabs Scribe v2 batch with keyterms
3. **uncertainty** — multi-signal confidence scoring (+ optional XGBoost risk signal)
4. **tavily_verify** — confidence-gated medical verification (cap=5/transcript, deduped, cached)
5. **claude_correct** — safe correction with hallucination guard
6. **claude_extract** — clinical entity extraction
7. **learning_loop** — phonetic map + correction history + adaptive keyterms (in-memory store, Redis-compatible API)

The hallucination guard in `claude_correct.py` reverts any "changed" word that
isn't backed by a Tavily-verified canonical match — this is what powers the
"0% unsafe guess rate" headline metric.

## Storage

Default storage is in-memory (`InMemoryStore` in `app/storage.py`), which mirrors a
small Redis subset (string TTL, hash, sorted set) for straightforward Redis swap-in.

## Tests

```bash
cd backend
pytest -q
```

`tests/test_claude_correct.py` is the most important — it locks in the
hallucination-guard invariant.
