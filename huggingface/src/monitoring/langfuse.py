"""Langfuse helper for tracing."""

from functools import lru_cache

from langfuse.callback import CallbackHandler

from src.config.settings import get_settings


@lru_cache(maxsize=1)
def get_langfuse_handler() -> CallbackHandler | None:
    settings = get_settings()
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return None
    return CallbackHandler(
        public_key=settings.langfuse_public_key.get_secret_value(),
        secret_key=settings.langfuse_secret_key.get_secret_value(),
        host=str(settings.langfuse_host) if settings.langfuse_host else "https://cloud.langfuse.com",
    )

