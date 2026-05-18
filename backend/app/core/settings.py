from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Dynamic AI Customer Support"
    environment: str = "dev"
    api_prefix: str = "/api/v1"

    secret_key: str = "change-this-in-env"
    jwt_algorithm: str = "HS256"
    access_token_ttl_minutes: int = 30
    refresh_token_ttl_minutes: int = 60 * 24 * 7

    redis_url: str = "redis://localhost:6379/0"
    redis_enabled: bool = True
    cache_default_ttl_seconds: int = 300
    session_ttl_seconds: int = 1800

    rate_limit: str = "60/minute"
    max_body_size_bytes: int = 1024 * 1024

    data_path: Path = Path(__file__).resolve().parents[1] / "data" / "training_data.txt"
    vector_artifact_path: Path = (
        Path(__file__).resolve().parents[1] / "data" / "vector_artifacts.json"
    )

    default_admin_username: str = "admin"
    default_admin_password: str = "admin123"
    default_admin_role: str = "admin"

    llm_backend: str = "extractive"
    llm_timeout_seconds: int = 20
    openai_compatible_url: str | None = None
    openai_compatible_api_key: str | None = None

    vector_backend: str = "inmemory"

    sentry_dsn: str | None = None
    cors_origins: str = "*"

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, value: str) -> str:
        allowed = {"dev", "staging", "prod"}
        normalized = value.lower().strip()
        if normalized not in allowed:
            raise ValueError(f"environment must be one of {sorted(allowed)}")
        return normalized

    @field_validator("secret_key")
    @classmethod
    def validate_secret_key(cls, value: str) -> str:
        if len(value.strip()) < 32:
            raise ValueError("secret_key must be at least 32 characters")
        return value

    @property
    def cors_origin_list(self) -> list[str]:
        value = self.cors_origins.strip()
        if value == "*":
            return ["*"]
        return [item.strip() for item in value.split(",") if item.strip()]

    def validate_startup_paths(self) -> None:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path does not exist: {self.data_path}")
        self.vector_artifact_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.validate_startup_paths()
    return settings
