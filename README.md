# CareCaller AI — Backend

FastAPI backend for the CareCaller AI verification-augmented STT pipeline. Implements
**Person B (AI/NLP)** and **Person C (Backend)** end-to-end. Person A's modules
(audio preprocessing, ElevenLabs Scribe v2, uncertainty scoring, keyterms,
benchmark generation) live as stubs — see `HANDOFF_PERSON_A.md`.

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
2. **Hand `HANDOFF_PERSON_A.md` to Person A.** They implement 5 functions and
   produce one JSON file. Until then, `/transcribe` returns 501 with the missing
   module's name in the body — everything else (`/health`, `/benchmark` 503) works.

## Endpoints

| Method | Path | Notes |
|---|---|---|
| POST | `/transcribe` | multipart `file=<audio>`. Returns `TranscribeResponse`. 501 until Person A's stubs are filled in. |
| GET  | `/benchmark?clips=all\|adversarial\|standard` | Serves `data/benchmark_results.json`. 503 if Person A hasn't generated it yet. |
| GET  | `/health` | Reachability check for Tavily + Claude. `redis: "in-memory"` (no Redis in this build). |

## Architecture (7 layers)

1. **preprocessing** — ffmpeg loudnorm + denoise [Person A stub]
2. **scribe** — ElevenLabs Scribe v2 batch with keyterms [Person A stub]
3. **uncertainty** — multi-signal confidence scoring [Person A stub]
4. **tavily_verify** — confidence-gated medical verification (cap=5/transcript, deduped, cached)
5. **claude_correct** — safe correction with hallucination guard
6. **claude_extract** — clinical entity extraction
7. **learning_loop** — phonetic map + correction history + adaptive keyterms (in-memory)

The hallucination guard in `claude_correct.py` reverts any "changed" word that
isn't backed by a Tavily-verified canonical match — this is what powers the
"0% unsafe guess rate" headline metric.

## Storage

In-memory only. `InMemoryStore` in `app/storage.py` mimics a small Redis subset
(string TTL, hash, sorted set). Single-process, resets on restart. Acceptable for
the hackathon demo where the "improves with each call" pitch only needs
within-process memory.

## Tests

```bash
cd backend
pytest -q
```

`tests/test_claude_correct.py` is the most important — it locks in the
hallucination-guard invariant.
