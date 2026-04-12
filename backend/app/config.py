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
    # Backward-compatible alias (some env files use ELEVEN_LABS_API_KEY)
    ELEVEN_LABS_API_KEY: SecretStr = SecretStr("")

    # Tunables
    CLAUDE_MODEL: str = "claude-sonnet-4-5"
    TAVILY_CALL_CAP: int = 5
    TAVILY_CACHE_TTL_SEC: int = 3600
    FRONTEND_ORIGIN: str = "http://localhost:5173"
    STREAM_TOKEN_TTL_SEC: int = 60
    SCRIBE_REALTIME_MODEL_ID: str = "scribe_v2_realtime"
    SCRIBE_REALTIME_WS_URL: str = "wss://api.elevenlabs.io/v1/speech-to-text/realtime"
    XGBOOST_MODEL_PATH: Path = BACKEND_DIR / "audio_gen" / "xgboost_telephony_model.joblib"
    XGBOOST_LOW_THRESHOLD: float = 0.60
    XGBOOST_MEDIUM_THRESHOLD: float = 0.35

    BENCHMARK_RESULTS_PATH: Path = BACKEND_DIR / "data" / "benchmark_results.json"

    def elevenlabs_api_key(self) -> str:
        primary = self.ELEVENLABS_API_KEY.get_secret_value().strip()
        if primary:
            return primary
        return self.ELEVEN_LABS_API_KEY.get_secret_value().strip()


settings = Settings()
