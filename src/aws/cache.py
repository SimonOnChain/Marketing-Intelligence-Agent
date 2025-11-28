"""Query caching using DynamoDB for persistent cache storage."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any

import boto3
from botocore.exceptions import ClientError

from src.config.settings import get_settings


@dataclass
class CacheEntry:
    """Represents a cached query result."""
    query_hash: str
    query: str
    answer: str
    sources: list[dict]
    agents_used: list[str]
    created_at: int
    ttl: int
    hit_count: int = 0


class QueryCache:
    """DynamoDB-backed query cache for fast repeated query responses."""

    TABLE_NAME = "marketing_agent_cache"
    DEFAULT_TTL_SECONDS = 3600 * 24  # 24 hours

    def __init__(self, use_local: bool = False):
        """Initialize DynamoDB cache client.

        Args:
            use_local: If True, use local DynamoDB (for development)
        """
        settings = get_settings()

        if use_local:
            self.dynamodb = boto3.resource(
                "dynamodb",
                endpoint_url="http://localhost:8000",
                region_name="us-east-1",
                aws_access_key_id="dummy",
                aws_secret_access_key="dummy",
            )
        else:
            self.dynamodb = boto3.resource(
                "dynamodb",
                region_name=settings.aws_region,
                aws_access_key_id=settings.aws_access_key_id.get_secret_value() if settings.aws_access_key_id else None,
                aws_secret_access_key=settings.aws_secret_access_key.get_secret_value() if settings.aws_secret_access_key else None,
            )

        self.table = None
        self._ensure_table_exists()

    def _ensure_table_exists(self) -> None:
        """Create the cache table if it doesn't exist."""
        try:
            self.table = self.dynamodb.Table(self.TABLE_NAME)
            self.table.load()
        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceNotFoundException":
                self._create_table()
            else:
                raise

    def _create_table(self) -> None:
        """Create the DynamoDB table for caching."""
        self.table = self.dynamodb.create_table(
            TableName=self.TABLE_NAME,
            KeySchema=[
                {"AttributeName": "query_hash", "KeyType": "HASH"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "query_hash", "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )
        self.table.wait_until_exists()

    @staticmethod
    def _hash_query(query: str) -> str:
        """Create a hash of the query for cache key."""
        normalized = query.lower().strip()
        return hashlib.sha256(normalized.encode()).hexdigest()[:32]

    def get(self, query: str) -> dict[str, Any] | None:
        """Retrieve cached result for a query.

        Args:
            query: The user's query string

        Returns:
            Cached result dict or None if not found/expired
        """
        if not self.table:
            return None

        query_hash = self._hash_query(query)

        try:
            response = self.table.get_item(Key={"query_hash": query_hash})

            if "Item" not in response:
                return None

            item = response["Item"]
            current_time = int(time.time())

            # Check TTL
            if item.get("ttl", 0) < current_time:
                self.delete(query)
                return None

            # Update hit count
            self.table.update_item(
                Key={"query_hash": query_hash},
                UpdateExpression="SET hit_count = hit_count + :inc",
                ExpressionAttributeValues={":inc": 1},
            )

            # Parse chart_data if present
            chart_data = None
            if item.get("chart_data"):
                try:
                    chart_data = json.loads(item["chart_data"])
                except (json.JSONDecodeError, TypeError):
                    pass

            return {
                "answer": item["answer"],
                "sources": json.loads(item.get("sources", "[]")),
                "agents_used": json.loads(item.get("agents_used", "[]")),
                "chart_data": chart_data,
                "cached": True,
                "cache_hit_count": item.get("hit_count", 0) + 1,
            }

        except ClientError:
            return None

    def set(
        self,
        query: str,
        answer: str,
        sources: list[dict],
        agents_used: list[str],
        chart_data: dict | None = None,
        ttl_seconds: int | None = None,
    ) -> bool:
        """Store a query result in the cache.

        Args:
            query: The user's query string
            answer: The generated answer
            sources: List of source documents
            agents_used: List of agents that were used
            chart_data: Optional chart data for visualization
            ttl_seconds: Time-to-live in seconds (default: 24 hours)

        Returns:
            True if successfully cached
        """
        if not self.table:
            return False

        query_hash = self._hash_query(query)
        current_time = int(time.time())
        ttl = current_time + (ttl_seconds or self.DEFAULT_TTL_SECONDS)

        try:
            item = {
                "query_hash": query_hash,
                "query": query,
                "answer": answer,
                "sources": json.dumps(sources),
                "agents_used": json.dumps(agents_used),
                "created_at": current_time,
                "ttl": ttl,
                "hit_count": 0,
            }
            # Add chart_data if present
            if chart_data:
                item["chart_data"] = json.dumps(chart_data)

            self.table.put_item(Item=item)
            return True
        except ClientError:
            return False

    def delete(self, query: str) -> bool:
        """Remove a query from the cache."""
        if not self.table:
            return False

        query_hash = self._hash_query(query)

        try:
            self.table.delete_item(Key={"query_hash": query_hash})
            return True
        except ClientError:
            return False

    def clear_all(self) -> int:
        """Clear all cached entries. Returns count of deleted items."""
        if not self.table:
            return 0

        deleted = 0
        try:
            response = self.table.scan()
            with self.table.batch_writer() as batch:
                for item in response.get("Items", []):
                    batch.delete_item(Key={"query_hash": item["query_hash"]})
                    deleted += 1
            return deleted
        except ClientError:
            return deleted

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        if not self.table:
            return {"enabled": False}

        try:
            response = self.table.scan(Select="COUNT")
            return {
                "enabled": True,
                "total_entries": response.get("Count", 0),
                "table_name": self.TABLE_NAME,
            }
        except ClientError:
            return {"enabled": False, "error": "Could not get stats"}


class InMemoryCache:
    """Simple in-memory cache for development/testing without AWS."""

    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600):
        self._cache: dict[str, dict] = {}
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds

    @staticmethod
    def _hash_query(query: str) -> str:
        normalized = query.lower().strip()
        return hashlib.sha256(normalized.encode()).hexdigest()[:32]

    def get(self, query: str) -> dict[str, Any] | None:
        query_hash = self._hash_query(query)

        if query_hash not in self._cache:
            return None

        entry = self._cache[query_hash]

        if entry["ttl"] < time.time():
            del self._cache[query_hash]
            return None

        entry["hit_count"] = entry.get("hit_count", 0) + 1

        return {
            "answer": entry["answer"],
            "sources": entry["sources"],
            "agents_used": entry["agents_used"],
            "chart_data": entry.get("chart_data"),
            "cached": True,
            "cache_hit_count": entry["hit_count"],
        }

    def set(
        self,
        query: str,
        answer: str,
        sources: list[dict],
        agents_used: list[str],
        chart_data: dict | None = None,
        ttl_seconds: int | None = None,
    ) -> bool:
        # Evict oldest if at capacity
        if len(self._cache) >= self._max_size:
            oldest_key = min(self._cache, key=lambda k: self._cache[k]["created_at"])
            del self._cache[oldest_key]

        query_hash = self._hash_query(query)
        self._cache[query_hash] = {
            "query": query,
            "answer": answer,
            "sources": sources,
            "agents_used": agents_used,
            "chart_data": chart_data,
            "created_at": time.time(),
            "ttl": time.time() + (ttl_seconds or self._ttl_seconds),
            "hit_count": 0,
        }
        return True

    def delete(self, query: str) -> bool:
        query_hash = self._hash_query(query)
        if query_hash in self._cache:
            del self._cache[query_hash]
            return True
        return False

    def clear_all(self) -> int:
        count = len(self._cache)
        self._cache.clear()
        return count

    def get_stats(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "total_entries": len(self._cache),
            "max_size": self._max_size,
            "type": "in_memory",
        }


def get_cache(use_dynamodb: bool = True) -> QueryCache | InMemoryCache:
    """Factory function to get appropriate cache implementation."""
    settings = get_settings()

    if use_dynamodb and settings.aws_access_key_id:
        try:
            return QueryCache()
        except Exception:
            pass

    return InMemoryCache()
