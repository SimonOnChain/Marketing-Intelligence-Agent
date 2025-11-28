"""
Redis/Valkey cache for fast query caching.
Uses AWS ElastiCache Serverless with TLS.
"""

import json
import hashlib
import logging
from typing import Any
import redis

logger = logging.getLogger(__name__)


class RedisCache:
    """Fast Redis/Valkey cache for LLM responses and embeddings."""

    def __init__(self, host: str, port: int = 6379, ssl: bool = True):
        self.host = host
        self.port = port
        self.enabled = False
        self.client = None

        try:
            self.client = redis.Redis(
                host=host,
                port=port,
                ssl=ssl,
                ssl_cert_reqs=None,  # For AWS ElastiCache
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
            )
            # Test connection
            self.client.ping()
            self.enabled = True
            logger.info(f"Redis cache connected: {host}:{port}")
        except Exception as e:
            logger.warning(f"Redis cache not available: {e}")
            self.enabled = False

    def _make_key(self, prefix: str, data: str) -> str:
        """Create a cache key from prefix and data."""
        hash_val = hashlib.md5(data.encode()).hexdigest()[:16]
        return f"{prefix}:{hash_val}"

    def get(self, key: str) -> Any | None:
        """Get value from cache."""
        if not self.enabled:
            return None
        try:
            value = self.client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.warning(f"Redis get error: {e}")
            return None

    def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set value in cache with TTL (default 1 hour)."""
        if not self.enabled:
            return False
        try:
            self.client.setex(key, ttl, json.dumps(value))
            return True
        except Exception as e:
            logger.warning(f"Redis set error: {e}")
            return False

    def get_llm_response(self, query: str) -> dict | None:
        """Get cached LLM response."""
        key = self._make_key("llm", query.lower().strip())
        return self.get(key)

    def set_llm_response(self, query: str, response: dict, ttl: int = 3600) -> bool:
        """Cache LLM response."""
        key = self._make_key("llm", query.lower().strip())
        return self.set(key, response, ttl)

    def get_embedding(self, text: str) -> list | None:
        """Get cached embedding."""
        key = self._make_key("emb", text.lower().strip())
        return self.get(key)

    def set_embedding(self, text: str, embedding: list, ttl: int = 86400) -> bool:
        """Cache embedding (default 24 hours)."""
        key = self._make_key("emb", text.lower().strip())
        return self.set(key, embedding, ttl)

    def get_rag_results(self, query: str) -> list | None:
        """Get cached RAG retrieval results."""
        key = self._make_key("rag", query.lower().strip())
        return self.get(key)

    def set_rag_results(self, query: str, results: list, ttl: int = 1800) -> bool:
        """Cache RAG results (default 30 minutes)."""
        key = self._make_key("rag", query.lower().strip())
        return self.set(key, results, ttl)

    def get_stats(self) -> dict:
        """Get cache statistics."""
        if not self.enabled:
            return {"enabled": False}
        try:
            info = self.client.info("stats")
            return {
                "enabled": True,
                "host": self.host,
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
                "keys": self.client.dbsize(),
            }
        except Exception as e:
            return {"enabled": True, "error": str(e)}

    def clear(self) -> bool:
        """Clear all cache entries."""
        if not self.enabled:
            return False
        try:
            self.client.flushdb()
            return True
        except Exception as e:
            logger.error(f"Redis clear error: {e}")
            return False


# Singleton instance
_redis_cache: RedisCache | None = None


def get_redis_cache() -> RedisCache | None:
    """Get or create Redis cache instance."""
    global _redis_cache

    if _redis_cache is not None:
        return _redis_cache

    from src.config.settings import get_settings
    settings = get_settings()

    redis_host = getattr(settings, 'redis_host', None)
    if not redis_host:
        return None

    redis_port = getattr(settings, 'redis_port', 6379)
    redis_ssl = getattr(settings, 'redis_ssl', True)

    _redis_cache = RedisCache(host=redis_host, port=redis_port, ssl=redis_ssl)
    return _redis_cache
