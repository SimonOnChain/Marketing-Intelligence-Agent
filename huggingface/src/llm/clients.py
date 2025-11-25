"""Utility functions for provisioning LLM clients."""

from functools import lru_cache
from typing import Any, Dict

from groq import Groq
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config.settings import get_settings


@lru_cache(maxsize=1)
def get_groq_client() -> Groq:
    settings = get_settings()
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY missing. Populate .env before running.")
    return Groq(api_key=settings.groq_api_key.get_secret_value())


@lru_cache(maxsize=1)
def get_xai_client() -> OpenAI:
    settings = get_settings()
    if not settings.xai_api_key:
        raise RuntimeError("XAI_API_KEY missing. Populate .env before running.")
    return OpenAI(
        base_url="https://api.x.ai/v1",
        api_key=settings.xai_api_key.get_secret_value(),
    )


def default_reasoning_model() -> str:
    """Return the production reasoning model ID."""

    return "grok-4-1-fast-reasoning"


@retry(wait=wait_exponential(min=1, max=20), stop=stop_after_attempt(5))
def call_xai_chat(messages: list[Dict[str, Any]], **kwargs: Any) -> str:
    """Thin wrapper with retry+parsing for chat responses."""

    client = get_xai_client()
    response = client.chat.completions.create(model=default_reasoning_model(), messages=messages, **kwargs)
    return response.choices[0].message.content or ""

