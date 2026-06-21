"""Application configuration.

Settings are loaded from environment variables and an optional ``.env`` file.
No secret ever appears as a hardcoded default. Sensible non-secret defaults let
the app boot for local development while clearly degrading features that need
credentials (LLM, flight/hotel providers) until those are supplied.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# config.py lives at backend/app/config.py
BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent
DATA_DIR = BACKEND_DIR / "data"

_DEFAULT_SQLITE_URL = f"sqlite:///{(DATA_DIR / 'flight_assistant.db').as_posix()}"
_DEFAULT_SEED_PATH = (DATA_DIR / "coupons.seed.json").as_posix()


class Settings(BaseSettings):
    """Strongly-typed application settings.

    Environment variable names map case-insensitively to field names (e.g.
    ``GROQ_API_KEY`` -> ``groq_api_key``).
    """

    model_config = SettingsConfigDict(
        env_file=(str(PROJECT_ROOT / ".env"), str(BACKEND_DIR / ".env"), ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Application ---
    app_name: str = "Flight & Travel Assistant"
    app_version: str = "1.0.0"
    app_env: str = "development"
    debug: bool = False
    log_level: str = "INFO"
    host: str = "0.0.0.0"
    port: int = 8000

    # --- LLM (Groq) ---
    groq_api_key: str | None = None
    groq_model: str = "llama-3.3-70b-versatile"
    groq_temperature: float = 0.2
    groq_max_tokens: int = 1024
    groq_timeout_seconds: float = 30.0

    # --- Flight / hotel provider (RapidAPI Sky Scrapper) ---
    rapidapi_key: str | None = None
    rapidapi_host: str = "sky-scrapper.p.rapidapi.com"

    # --- HTTP client behaviour ---
    http_timeout_seconds: float = 20.0
    http_max_retries: int = 2

    # --- Database ---
    database_url: str = _DEFAULT_SQLITE_URL

    # --- Localisation defaults ---
    default_currency: str = "INR"
    default_market: str = "en-US"
    default_country: str = "IN"

    # --- Coupons ---
    coupon_seed_path: str = _DEFAULT_SEED_PATH
    coupon_sources: str = ""  # comma-separated URLs
    coupon_use_playwright: bool = False
    coupon_refresh_enabled: bool = True
    coupon_refresh_interval_minutes: int = 360

    # --- Scheduler ---
    enable_scheduler: bool = True

    # --- API ---
    cors_allow_origins: str = "*"  # comma-separated origins
    max_results: int = 10

    @property
    def cors_allow_origins_list(self) -> list[str]:
        """Parse the CORS origins string into a list."""
        return self._split_csv(self.cors_allow_origins) or ["*"]

    @property
    def coupon_sources_list(self) -> list[str]:
        """Parse coupon source URLs into a list."""
        return self._split_csv(self.coupon_sources)

    @property
    def is_llm_configured(self) -> bool:
        return bool(self.groq_api_key)

    @property
    def is_provider_configured(self) -> bool:
        return bool(self.rapidapi_key)

    @staticmethod
    def _split_csv(value: str) -> list[str]:
        return [item.strip() for item in value.split(",") if item.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance."""
    return Settings()
