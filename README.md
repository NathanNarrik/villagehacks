# CareCaller AI — Backend

FastAPI backend for the CareCaller AI verification-augmented STT pipeline. Implements
**Person B (AI/NLP)** and **Person C (Backend)** end-to-end. Person A's modules
(audio preprocessing, ElevenLabs Scribe v2, uncertainty scoring, keyterms,
benchmark generation) live as stubs — see `HANDOFF_PERSON_A.md`.

## Full stack (frontend + backend)

Run the API and the Vite app in **two terminals**. The UI is configured for `http://localhost:8000` via `frontend/.env` (`VITE_API_URL`).

**Terminal 1 — backend**

The app reads secrets from **`backend/.env`**, not `.env.example`. Copy the example once, then edit the copy:

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS/Linux
pip install -r requirements.txt
copy .env.example .env             # Windows — creates backend/.env
# cp .env.example .env             # macOS/Linux
# Edit .env and paste ANTHROPIC_API_KEY, TAVILY_API_KEY, ELEVENLABS_API_KEY
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

**Windows: `WinError 10013` when starting Uvicorn** — Windows often blocks or reserves port **8000** (Hyper-V, WSL, Docker, etc.). Use another port and bind to loopback:

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8001
```

Then set `VITE_API_URL=http://127.0.0.1:8001` in `frontend/.env` and restart `npm run dev`. To see excluded ranges: `netsh interface ipv4 show excludedportrange protocol=tcp` (PowerShell as Admin).

**Terminal 2 — frontend**

From the **repository root** (or from `frontend/`):

```bash
cd frontend
npm install
npm run dev
```

Or from the repo root: `npm install --prefix frontend` once, then `npm run dev` (root `package.json` forwards to the frontend).

Open the **Local** URL Vite prints (e.g. `http://localhost:8080`, or `8081` if 8080 is busy). Prefer that over the **Network** IP URL unless you extend CORS for LAN origins. The API allows `http://localhost` / `127.0.0.1` on **any port**.

**Failed fetches in the browser?** (1) Backend must be running first, on the **same host/port** as `VITE_API_URL` (default `http://localhost:8000`). (2) After editing `frontend/.env`, **restart** `npm run dev`. (3) Open the app via the **Local** `http://localhost:…` URL Vite printed, not the **Network** `192.168…` URL, unless you extend CORS.

**If `npm run dev` fails:** run it from `frontend/` or the repo root only (not a parent folder). On Windows, if the dev server still exits immediately, try `set VITE_DISABLE_LOVABLE_TAGGER=1` then `npm run dev` to disable the Lovable component tagger plugin. The Browserslist “caniuse-lite is old” message is a warning, not a failure.

- **Demo**: each scenario loads a **`.wav`** from `frontend/public/demo-audio/` (same basename as the scenario id), sends it to `POST /transcribe`, and renders the real response. Replace those clips with your own recordings as needed; the repo ships short placeholder WAVs so paths resolve out of the box.
- **Benchmark**: loads `GET /benchmark` when the backend is up; falls back to embedded sample data if the server is down or returns an error. The file `backend/data/benchmark_results.json` is gitignored — copy `backend/data/benchmark_results.json.example` to that path (or run Person A’s benchmark script) so the API can serve real results instead of 503.

## Backend only

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS/Linux
pip install -r requirements.txt
copy .env.example .env             # then fill in API keys
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Point any client at the same URL you pass to Uvicorn (e.g. `http://127.0.0.1:8000`; set `VITE_API_URL` in the frontend to match).

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
