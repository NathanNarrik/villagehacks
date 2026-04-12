# CareCaller AI — Backend

FastAPI backend for the CareCaller AI verification-augmented STT pipeline.

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

**Windows: `ffmpeg` / `ffprobe` not found** — Preprocessing shells out to **ffmpeg** and **ffprobe**. Install them and ensure they are on `PATH`, then **restart the terminal** (and Uvicorn).

```powershell
winget install --id Gyan.FFmpeg -e
```

After install, confirm in a **new** PowerShell window: `ffmpeg -version` and `ffprobe -version`. If `winget` is unavailable, use [ffmpeg.org](https://ffmpeg.org/download.html) builds and add the `bin` folder to your user or system PATH.

If the error persists while running Uvicorn **from Cursor/VS Code**, the IDE often inherits an old `PATH`. The backend now also searches common Windows install dirs (including WinGet’s `Links` folder). As a last resort, set **`FFMPEG_PATH`** (and optionally **`FFPROBE_PATH`**) in `backend/.env` to the full paths of `ffmpeg.exe` / `ffprobe.exe` — see `backend/.env.example`.

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
7. **learning_loop** — phonetic map + correction history + adaptive keyterms (in-memory store)

The hallucination guard in `claude_correct.py` reverts any "changed" word that
isn't backed by a Tavily-verified canonical match — this is what powers the
"0% unsafe guess rate" headline metric.

## Storage

Default storage is in-memory (`InMemoryStore` in `app/storage.py`) with string TTL,
hash, and sorted-set primitives used by the learning loop and cache.

## Tests

```bash
cd backend
pytest -q
```

`tests/test_claude_correct.py` is the most important — it locks in the
hallucination-guard invariant.
