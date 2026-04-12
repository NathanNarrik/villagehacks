"""Environment helpers shared by audio generation entrypoints."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_dotenv_if_present(path: Path) -> None:
    """Load a dotenv-style file into the process environment without overriding."""

    if not path.exists() or not path.is_file():
        return

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def load_audio_gen_env(paths: Sequence[Path] | None = None) -> None:
    """Load standard repo dotenv files used by backend audio tooling."""

    candidates = list(paths or (REPO_ROOT / ".env", REPO_ROOT / "backend" / ".env"))
    for path in candidates:
        load_dotenv_if_present(path)


def resolve_elevenlabs_api_key(env: Mapping[str, str] | None = None) -> str:
    """Accept the current primary env var and the legacy alias."""

    source = env or os.environ
    primary = source.get("ELEVENLABS_API_KEY", "").strip()
    if primary:
        return primary
    return source.get("ELEVEN_LABS_API_KEY", "").strip()
