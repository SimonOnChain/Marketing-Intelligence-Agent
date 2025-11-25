"""Centralized configuration via pydantic-settings."""

from functools import lru_cache
from typing import Literal

from pydantic import AnyUrl, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False)

    env: Literal["development", "staging", "production"] = Field(default="development")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO", alias="LOG_LEVEL")

    # LLM providers
    groq_api_key: SecretStr | None = Field(default=None, alias="GROQ_API_KEY")
    xai_api_key: SecretStr | None = Field(default=None, alias="XAI_API_KEY")

    # Vector DB
    qdrant_url: AnyUrl | None = Field(default=None, alias="QDRANT_URL")
    qdrant_api_key: SecretStr | None = Field(default=None, alias="QDRANT_API_KEY")
    qdrant_collection: str = Field(default="reviews")

    # Monitoring
    langfuse_public_key: SecretStr | None = Field(default=None, alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: SecretStr | None = Field(default=None, alias="LANGFUSE_SECRET_KEY")
    langfuse_host: AnyUrl | None = Field(default="https://cloud.langfuse.com", alias="LANGFUSE_HOST")

    # AWS
    aws_access_key_id: SecretStr | None = Field(default=None, alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: SecretStr | None = Field(default=None, alias="AWS_SECRET_ACCESS_KEY")
    aws_region: str = Field(default="eu-central-1", alias="AWS_REGION")

    # Data paths
    data_raw_dir: str = Field(default="data/raw")
    data_processed_dir: str = Field(default="data/processed")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached settings instance."""

    return Settings()

