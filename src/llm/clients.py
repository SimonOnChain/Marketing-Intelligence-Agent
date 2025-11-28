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
    return "grok-3-fast"  # Faster than grok-4-1-fast-reasoning


def fast_model() -> str:
    """Return the fastest model for simple tasks."""
    return "grok-3-fast"


@retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3))
def call_xai_chat(
    messages: list[Dict[str, Any]],
    model: str = None,
    return_usage: bool = False,
    **kwargs: Any
) -> str | tuple[str, dict]:
    """Thin wrapper with retry+parsing for chat responses.

    Args:
        messages: List of message dicts with role and content
        model: Model to use (defaults to reasoning model)
        return_usage: If True, returns (content, usage_dict) tuple
        **kwargs: Additional args for the API call

    Returns:
        str if return_usage=False, else (str, usage_dict)
    """
    client = get_xai_client()
    model_to_use = model or default_reasoning_model()
    response = client.chat.completions.create(model=model_to_use, messages=messages, **kwargs)
    content = response.choices[0].message.content or ""

    if return_usage:
        usage = {}
        if response.usage:
            usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        return content, usage

    return content


# Simple in-memory cache for repeated queries
_response_cache: Dict[str, str] = {}
_cache_max_size = 100


def call_xai_chat_cached(messages: list[Dict[str, Any]], **kwargs: Any) -> tuple[str, bool]:
    """Call xAI with caching. Returns (response, was_cached)."""
    # Create cache key from messages
    cache_key = str(messages) + str(kwargs)

    if cache_key in _response_cache:
        return _response_cache[cache_key], True

    response = call_xai_chat(messages, **kwargs)

    # Store in cache (with size limit)
    if len(_response_cache) >= _cache_max_size:
        # Remove oldest entry
        _response_cache.pop(next(iter(_response_cache)))
    _response_cache[cache_key] = response

    return response, False

