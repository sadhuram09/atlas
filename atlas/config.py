"""
atlas/config.py

All configuration loaded from environment variables.
pydantic-settings gives us:
  - Type coercion (str → int, str → bool)
  - Validation at startup (fail fast if GROQ_API_KEY is missing)
  - .env file support in local dev
  - IDE autocompletion everywhere

LLM provider: Groq (free tier, no credit card required)
  - API is OpenAI-compatible → we use the openai SDK pointed at Groq
  - Get your key at: https://console.groq.com/keys

Usage:
    from atlas.config import settings
    client = AsyncOpenAI(api_key=settings.groq_key, base_url=settings.groq_base_url)
"""

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    ATLAS runtime configuration.

    All fields have defaults so the app starts without a .env file
    (useful for CI and cold-start testing). Production Railway deployment
    sets real values via the Railway dashboard → Variables tab.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Ignore unknown env vars (e.g. Railway injects PORT)
    )

    # ------------------------------------------------------------------
    # App identity
    # ------------------------------------------------------------------
    app_name: str = "ATLAS"
    app_version: str = "0.1.0"
    environment: str = Field(default="development", pattern="^(development|staging|production)$")
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")

    # ------------------------------------------------------------------
    # Server
    # ------------------------------------------------------------------
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1024, le=65535)

    # ------------------------------------------------------------------
    # LLM — Groq (free, no credit card)
    # Get your key at: https://console.groq.com/keys
    # ------------------------------------------------------------------
    groq_api_key: SecretStr = Field(default=SecretStr("gsk_placeholder"))
    groq_base_url: str = "https://api.groq.com/openai/v1"

    # Groq model tiers — all free on the generous free plan
    # FAST     → llama-3.1-8b-instant   (fastest, great for drafts)
    # BALANCED → llama-3.3-70b-versatile (best all-rounder, use this most)
    # POWERFUL → deepseek-r1-distill-llama-70b (reasoning tasks)
    model_fast: str = "llama-3.1-8b-instant"
    model_balanced: str = "llama-3.3-70b-versatile"
    model_powerful: str = "deepseek-r1-distill-llama-70b"

    # ------------------------------------------------------------------
    # Database (Phase 2+)
    # ------------------------------------------------------------------
    database_url: str = Field(
        default="postgresql+asyncpg://atlas:atlas@localhost:5432/atlas"
    )

    # ------------------------------------------------------------------
    # Docker sandbox (Phase 1+)
    # ------------------------------------------------------------------
    docker_timeout_seconds: int = Field(default=30, ge=5, le=300)
    docker_memory_limit: str = "256m"
    docker_network: str = "none"  # Isolated — no internet access in sandbox

    # ------------------------------------------------------------------
    # LangSmith observability (Phase 2+)
    # ------------------------------------------------------------------
    langsmith_api_key: SecretStr = Field(default=SecretStr("ls-placeholder"))
    langsmith_project: str = "atlas-designpro"
    langchain_tracing_v2: bool = Field(default=False)

    # ------------------------------------------------------------------
    # CORS — allow the Vercel frontend to call the Railway backend
    # ------------------------------------------------------------------
    cors_origins: list[str] = Field(
        default=[
            "http://localhost:5173",   # Vite dev server
            "http://localhost:3000",
            "https://designpro.app",   # Production Vercel domain
        ]
    )

    # ------------------------------------------------------------------
    # Task limits
    # ------------------------------------------------------------------
    max_concurrent_tasks: int = Field(default=5, ge=1, le=50)
    default_max_retries: int = Field(default=3, ge=1, le=10)
    global_budget_usd: float = Field(default=5.0, gt=0)

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def groq_key(self) -> str:
        """Unwrap the Groq secret for the OpenAI SDK client."""
        return self.groq_api_key.get_secret_value()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Cached settings singleton.

    The lru_cache means Settings is only instantiated once — not on
    every request. Call get_settings() everywhere; never Settings() directly.
    """
    return Settings()


# Convenience alias — import this everywhere
settings = get_settings()
