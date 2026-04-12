"""Environment-driven settings for the CareCaller backend."""
from __future__ import annotations

from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Person B
    ANTHROPIC_API_KEY: SecretStr = SecretStr("")
    TAVILY_API_KEY: SecretStr = SecretStr("")

    # Person A
    ELEVENLABS_API_KEY: SecretStr = SecretStr("")

    # Tunables
    CLAUDE_MODEL: str = "claude-sonnet-4-5"
    TAVILY_CALL_CAP: int = 5
    TAVILY_CACHE_TTL_SEC: int = 3600
    FRONTEND_ORIGIN: str = "http://localhost:5173"

    BENCHMARK_RESULTS_PATH: Path = BACKEND_DIR / "data" / "benchmark_results.json"


settings = Settings()
