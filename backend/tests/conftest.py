from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure imports work whether pytest is run from repo root or backend/
# - `app.*` imports require backend/ on sys.path
# - `backend.*` imports require repo root on sys.path
BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]

for path in (BACKEND_ROOT, REPO_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from app.claude_correct import reset_corrector
from app.storage import store
from app.tavily_verify import reset_verifier
from stt.runtime import reset_runtime_cache


@pytest.fixture(autouse=True)
def _reset_state():
    """Reset in-memory services between tests."""
    store.reset()
    reset_verifier()
    reset_corrector()
    reset_runtime_cache()
    yield
    store.reset()
    reset_verifier()
    reset_corrector()
    reset_runtime_cache()
