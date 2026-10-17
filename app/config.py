"""Environment-backed CareBridge configuration."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Validated settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "CareBridge"
    environment: str = "demo"

    # RUN_MODE=live  -> real model required; local fallback is forbidden.
    # RUN_MODE=demo  -> local extractive fallback allowed; DEMO MODE banner shown.
    run_mode: Literal["live", "demo"] = "demo"

    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str = "Qwen2.5-7B-Instruct"
    llm_timeout_seconds: float = Field(default=20.0, gt=0, le=180)
    # Per-stage timeouts (seconds); 0 means disabled (falls back to llm_timeout_seconds).
    llm_redflag_timeout_seconds: float = Field(default=4.0, ge=0, le=30)
    llm_generation_timeout_seconds: float = Field(default=15.0, ge=0, le=120)
    llm_grounding_timeout_seconds: float = Field(default=6.0, ge=0, le=30)
    llm_max_retries: int = Field(default=2, ge=0, le=5)
    # Circuit breaker: open after this many consecutive failures.
    llm_circuit_breaker_threshold: int = Field(default=5, ge=1, le=20)
    # Circuit breaker: seconds to stay open before a half-open probe.
    llm_circuit_breaker_cooldown: float = Field(default=30.0, ge=5, le=300)

    gate_threshold: float = Field(default=0.65, ge=0, le=1)
    gate_enabled: bool = True
    consistency_samples: int = Field(default=3, ge=1, le=3)
    embeddings_enabled: bool = True
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    database_url: str = "sqlite:///./data/carebridge.db"
    patients_file: Path = ROOT_DIR / "data" / "patients.json"
    docs_dir: Path = ROOT_DIR / "data" / "docs"
    redflags_dir: Path = ROOT_DIR / "data" / "redflags"
    gate_weights_file: Path = ROOT_DIR / "eval" / "gate_weights.json"
    nurse_password: str | None = None
    cors_origins: str = "http://localhost:8000"
    requests_per_minute: int = Field(default=20, ge=1, le=600)
    max_message_length: int = Field(default=1000, ge=1, le=1000)
    emergency_contact_text: str = "your locally verified emergency services"
    backend: str = "local"
    evolus_base_url: str | None = None
    evolus_api_key: str | None = None
    evolus_workspace_id: str | None = None
    amd_served: bool = False
    sqlite_volume_path: str = "./data"

    @property
    def allowed_origins(self) -> list[str]:
        """Return configured comma-separated CORS origins."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def is_live(self) -> bool:
        """True when RUN_MODE=live."""
        return self.run_mode == "live"


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings instance."""
    return Settings()
