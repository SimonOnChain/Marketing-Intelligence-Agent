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

    # AWS Core
    aws_access_key_id: SecretStr | None = Field(default=None, alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: SecretStr | None = Field(default=None, alias="AWS_SECRET_ACCESS_KEY")
    aws_region: str = Field(default="eu-central-1", alias="AWS_REGION")

    # AWS S3
    s3_bucket: str | None = Field(default=None, alias="S3_BUCKET")

    # AWS Cognito
    cognito_user_pool_id: str | None = Field(default=None, alias="COGNITO_USER_POOL_ID")
    cognito_client_id: str | None = Field(default=None, alias="COGNITO_CLIENT_ID")
    cognito_client_secret: SecretStr | None = Field(default=None, alias="COGNITO_CLIENT_SECRET")

    # AWS Bedrock
    bedrock_enabled: bool = Field(default=False, alias="BEDROCK_ENABLED")

    # Caching
    cache_enabled: bool = Field(default=True, alias="CACHE_ENABLED")
    cache_ttl_seconds: int = Field(default=86400, alias="CACHE_TTL_SECONDS")  # 24 hours
    use_dynamodb_cache: bool = Field(default=False, alias="USE_DYNAMODB_CACHE")

    # Redis/Valkey Cache
    redis_host: str | None = Field(default=None, alias="REDIS_HOST")
    redis_port: int = Field(default=6379, alias="REDIS_PORT")
    redis_ssl: bool = Field(default=True, alias="REDIS_SSL")
    use_redis_cache: bool = Field(default=False, alias="USE_REDIS_CACHE")

    # Feature Flags
    use_bedrock_for_intent: bool = Field(default=False, alias="USE_BEDROCK_FOR_INTENT")
    use_bedrock_for_synthesis: bool = Field(default=False, alias="USE_BEDROCK_FOR_SYNTHESIS")

    # Data paths
    data_raw_dir: str = Field(default="data/raw")
    data_processed_dir: str = Field(default="data/processed")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached settings instance."""

    return Settings()

