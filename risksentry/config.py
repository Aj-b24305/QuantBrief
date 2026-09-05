from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

LlmProvider = Literal["ollama", "openai", "gemini"]

#: OpenAI-compatible Gemini endpoint (Google exposes one under /v1beta/openai).
DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


class Settings(BaseSettings):
    """Runtime configuration.

    Every field can be overridden through environment variables (a local ``.env``
    file is also supported). The ``RISKSENTRY_`` prefix applies to all of them,
    e.g. ``RISKSENTRY_LLM_MODEL=qwen3.5:9b``. Provider-specific credentials
    (``GEMINI_API_KEY``, ``OLLAMA_TIMEOUT_SECONDS``) are also accepted without
    the prefix, so standard cloud/registry tooling keeps working.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="RISKSENTRY_",
        extra="ignore",
    )

    # --- Quantitative engine defaults -------------------------------------
    risk_free_rate: float = 0.0525  # annualized, e.g. 5.25%
    default_benchmark: str = "^NSEI"
    default_lookback_days: int = 365
    trading_days_per_year: int = 252
    var_confidence: float = 0.95
    min_observations: int = 30

    # --- LLM synthesizer (local, OpenAI-compatible) ------------------------
    llm_provider: LlmProvider = "ollama"
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str | None = None
    llm_timeout_seconds: float = 60.0  # generic (Gemini/cloud) timeout
    llm_max_retries: int = 1
    llm_temperature: float = 0.2

    # Cap on local Ollama HTTP requests (OLLAMA_TIMEOUT_SECONDS=120).
    ollama_timeout_seconds: float = Field(
        default=120.0,
        validation_alias=AliasChoices("RISKSENTRY_OLLAMA_TIMEOUT_SECONDS", "OLLAMA_TIMEOUT_SECONDS"),
    )

    # --- Gemini cloud fallback (OpenAI-compatible endpoint) ----------------
    gemini_base_url: str = Field(
        default=DEFAULT_GEMINI_BASE_URL,
        validation_alias=AliasChoices("GEMINI_BASE_URL", "RISKSENTRY_GEMINI_BASE_URL"),
    )
    gemini_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("GEMINI_API_KEY", "RISKSENTRY_GEMINI_API_KEY"),
    )
    gemini_model: str = Field(
        default="gemini-3.6-flash",
        validation_alias=AliasChoices("GEMINI_MODEL", "RISKSENTRY_GEMINI_MODEL"),
    )

    def resolve_llm_config(self) -> tuple[str, str | None, str]:
        """Resolve ``(base_url, api_key, model)`` for the configured provider.

        - ``ollama`` (default): http://localhost:11434/v1 + ``qwen3.5:9b``.
          Override the model with ``RISKSENTRY_LLM_MODEL`` (e.g. the lightweight
          ``deepseek-r1:1.5b`` alternative).
        - ``gemini``: OpenAI-compatible Gemini endpoint + ``gemini-2.0-flash``.
        - ``openai``: https://api.openai.com/v1 + ``gpt-4o-mini``.
        """
        provider = self.llm_provider.strip().lower()
        if provider == "gemini":
            base_url = self.llm_base_url or self.gemini_base_url
            api_key = self.llm_api_key or self.gemini_api_key
            model = self.llm_model or self.gemini_model
        elif provider == "openai":
            base_url = self.llm_base_url or "https://api.openai.com/v1"
            api_key = self.llm_api_key or self.gemini_api_key
            model = self.llm_model or "gpt-4o-mini"
        else:  # ollama (default) — local model
            base_url = self.llm_base_url or "http://localhost:11434/v1"
            api_key = self.llm_api_key or "ollama"
            model = self.llm_model or "qwen3.5:9b"
        return base_url, api_key, model

    def resolve_gemini_config(self) -> tuple[str, str | None, str]:
        """Resolve ``(base_url, api_key, model)`` for the Gemini cloud fallback."""
        return self.gemini_base_url, self.gemini_api_key, self.gemini_model


@lru_cache
def get_settings() -> Settings:
    """Process-wide settings singleton."""
    return Settings()
