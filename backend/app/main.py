"""FastAPI entry point for the CareCaller AI backend.

Three routes match the frozen frontend contract in `frontend/src/services/api.ts`:

    POST /transcribe   — accepts an audio UploadFile, runs the 7-layer pipeline
    GET  /benchmark    — returns the cached benchmark JSON Person A produces
    GET  /health       — reachability check for Tavily, Claude, and the (in-memory) store

Person A's modules raise NotImplementedError until they're wired up. We catch that
in `/transcribe` and return a clean 501 with the missing module name so debugging
is obvious.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal

from anthropic import AsyncAnthropic
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tavily import TavilyClient

from . import pipeline
from .config import settings
from .schemas import BenchmarkResponse, HealthResponse, TranscribeResponse

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("CareCaller backend starting (in-memory store, model=%s)", settings.CLAUDE_MODEL)
    yield
    log.info("CareCaller backend shutting down")


app = FastAPI(title="CareCaller AI Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_ORIGIN, "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# POST /transcribe
# ---------------------------------------------------------------------------
@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(file: UploadFile = File(...)) -> TranscribeResponse:
    suffix = Path(file.filename or "upload.wav").suffix or ".wav"
    tmp_path: str | None = None
    try:
        with NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        return await pipeline.run_full_pipeline(tmp_path)

    except NotImplementedError as exc:
        # A Person A stub was hit. Surface the missing function name in the body.
        raise HTTPException(
            status_code=501,
            detail=f"Person A module not implemented: {exc}",
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        log.exception("Pipeline failed")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# GET /benchmark
# ---------------------------------------------------------------------------
@app.get("/benchmark", response_model=BenchmarkResponse)
def benchmark(
    clips: Literal["all", "adversarial", "standard"] = Query("all"),
) -> BenchmarkResponse:
    path = settings.BENCHMARK_RESULTS_PATH
    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "Benchmark results not generated yet — Person A: run "
                "scripts/run_benchmark.py to produce data/benchmark_results.json"
            ),
        )

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(500, f"benchmark_results.json is malformed: {exc}")

    if clips != "all":
        wanted = "Adversarial" if clips == "adversarial" else "Standard"
        data["results"] = [
            r for r in data.get("results", []) if r.get("difficulty") == wanted
        ]

    return BenchmarkResponse.model_validate(data)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------
async def _ping_tavily() -> str:
    key = settings.TAVILY_API_KEY.get_secret_value()
    if not key:
        return "not_configured"
    try:
        client = TavilyClient(api_key=key)
        await asyncio.wait_for(
            asyncio.to_thread(
                client.search, query="acetaminophen", max_results=1, search_depth="basic"
            ),
            timeout=2.0,
        )
        return "reachable"
    except Exception as exc:  # noqa: BLE001
        return f"error: {type(exc).__name__}"


async def _ping_claude() -> str:
    key = settings.ANTHROPIC_API_KEY.get_secret_value()
    if not key:
        return "not_configured"
    try:
        client = AsyncAnthropic(api_key=key)
        await asyncio.wait_for(
            client.messages.create(
                model=settings.CLAUDE_MODEL,
                max_tokens=4,
                messages=[{"role": "user", "content": "ping"}],
            ),
            timeout=2.0,
        )
        return "reachable"
    except Exception as exc:  # noqa: BLE001
        return f"error: {type(exc).__name__}"


def _scribe_status() -> str:
    """Heuristic: scribe stub raises NotImplementedError on first call. We can't
    invoke it from /health (no audio), so we just check whether Person A's module
    still has the stub marker."""
    try:
        from . import scribe

        src = Path(scribe.__file__).read_text(encoding="utf-8")
        if "NotImplementedError" in src and "Person A" in src:
            return "stub"
        return "ready"
    except Exception:
        return "unknown"


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    tavily_status, claude_status = await asyncio.gather(
        _ping_tavily(), _ping_claude()
    )
    return HealthResponse(
        status="ok",
        redis="in-memory",
        scribe=_scribe_status(),
        tavily=tavily_status,
        claude=claude_status,
    )
