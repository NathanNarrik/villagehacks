"""Shared pytest fixtures: reset the in-memory store between tests."""
from __future__ import annotations

import pytest

from app.claude_correct import reset_corrector
from app.storage import store
from app.tavily_verify import reset_verifier


@pytest.fixture(autouse=True)
def _reset_state():
    store.reset()
    reset_verifier()
    reset_corrector()
    yield
    store.reset()
    reset_verifier()
    reset_corrector()
