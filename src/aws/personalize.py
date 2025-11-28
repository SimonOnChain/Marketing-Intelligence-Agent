"""Amazon Personalize integration for recommendations and personalization."""

from __future__ import annotations

import json
import hashlib
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError

from src.config.settings import get_settings

DATA_DIR = Path(__file__).parent.parent.parent / "data"
USER_INTERACTIONS_FILE = DATA_DIR / "feedback" / "user_interactions.jsonl"


class PersonalizeClient:
    """Client for Amazon Personalize recommendations.

    Provides both:
    1. Local collaborative filtering (fallback) for immediate results
    2. AWS Personalize integration for production-grade recommendations
    """

    def __init__(self):
        """Initialize Personalize client."""
        settings = get_settings()

        self.client = boto3.client(
            "personalize",
            region_name=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id.get_secret_value() if settings.aws_access_key_id else None,
            aws_secret_access_key=settings.aws_secret_access_key.get_secret_value() if settings.aws_secret_access_key else None,
        )

        self.runtime_client = boto3.client(
            "personalize-runtime",
            region_name=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id.get_secret_value() if settings.aws_access_key_id else None,
            aws_secret_access_key=settings.aws_secret_access_key.get_secret_value() if settings.aws_secret_access_key else None,
        )

        self._enabled = bool(settings.aws_access_key_id)
        self._interactions: list[dict] = []
        self._load_interactions()

    @property
    def enabled(self) -> bool:
        """Check if Personalize is available."""
        return self._enabled

    def _load_interactions(self):
        """Load user interactions from file."""
        USER_INTERACTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
        if USER_INTERACTIONS_FILE.exists():
            try:
                with open(USER_INTERACTIONS_FILE, "r", encoding="utf-8") as f:
                    self._interactions = [json.loads(line) for line in f if line.strip()]
            except Exception:
                self._interactions = []

    def _save_interaction(self, interaction: dict):
        """Save a user interaction."""
        with open(USER_INTERACTIONS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(interaction) + "\n")
        self._interactions.append(interaction)

    def record_interaction(
        self,
        user_id: str,
        item_id: str,
        interaction_type: str = "query",
        rating: float | None = None,
        metadata: dict | None = None
    ):
        """Record a user interaction for learning.

        Args:
            user_id: User identifier (can be session ID)
            item_id: Item identifier (query category, product, etc.)
            interaction_type: Type of interaction (query, view, purchase, like)
            rating: Optional rating (1-5)
            metadata: Additional metadata
        """
        interaction = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "item_id": item_id,
            "interaction_type": interaction_type,
            "rating": rating,
            "metadata": metadata or {},
        }
        self._save_interaction(interaction)

    def get_recommendations_local(
        self,
        user_id: str,
        num_results: int = 5,
        item_type: str = "query"
    ) -> list[dict]:
        """Get personalized recommendations using local collaborative filtering.

        Args:
            user_id: User identifier
            num_results: Number of recommendations to return
            item_type: Type of items to recommend

        Returns:
            List of recommended items with scores
        """
        if not self._interactions:
            return self._get_popular_items(num_results, item_type)

        # Get user's past interactions
        user_items = set()
        for interaction in self._interactions:
            if interaction["user_id"] == user_id:
                user_items.add(interaction["item_id"])

        # Find similar users (users who interacted with same items)
        similar_users = {}
        for interaction in self._interactions:
            if interaction["user_id"] != user_id and interaction["item_id"] in user_items:
                if interaction["user_id"] not in similar_users:
                    similar_users[interaction["user_id"]] = 0
                similar_users[interaction["user_id"]] += 1

        if not similar_users:
            return self._get_popular_items(num_results, item_type)

        # Get items from similar users that current user hasn't seen
        recommendations = {}
        for interaction in self._interactions:
            if interaction["user_id"] in similar_users and interaction["item_id"] not in user_items:
                item = interaction["item_id"]
                if item not in recommendations:
                    recommendations[item] = 0
                # Weight by user similarity
                recommendations[item] += similar_users[interaction["user_id"]]
                # Boost by rating if available
                if interaction.get("rating"):
                    recommendations[item] += interaction["rating"]

        # Sort by score
        sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)

        return [
            {"item_id": item, "score": score, "source": "collaborative"}
            for item, score in sorted_recs[:num_results]
        ]

    def _get_popular_items(self, num_results: int, item_type: str) -> list[dict]:
        """Get most popular items as fallback."""
        item_counts = {}
        for interaction in self._interactions:
            item = interaction["item_id"]
            if item not in item_counts:
                item_counts[item] = 0
            item_counts[item] += 1

        sorted_items = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)

        return [
            {"item_id": item, "score": count, "source": "popular"}
            for item, count in sorted_items[:num_results]
        ]

    def get_similar_queries(self, query: str, num_results: int = 3) -> list[str]:
        """Get similar queries that worked well.

        Args:
            query: Current query
            num_results: Number of suggestions

        Returns:
            List of similar successful queries
        """
        # Get queries with positive ratings
        positive_queries = [
            i["item_id"] for i in self._interactions
            if i.get("rating", 0) >= 4 or i.get("interaction_type") == "like"
        ]

        if not positive_queries:
            return []

        # Simple keyword matching for similarity
        query_words = set(query.lower().split())
        scored_queries = []

        for pq in positive_queries:
            pq_words = set(pq.lower().split())
            overlap = len(query_words & pq_words)
            if overlap > 0 and pq.lower() != query.lower():
                scored_queries.append((pq, overlap))

        # Sort by overlap score
        scored_queries.sort(key=lambda x: x[1], reverse=True)

        # Return unique queries
        seen = set()
        results = []
        for q, _ in scored_queries:
            if q not in seen:
                seen.add(q)
                results.append(q)
                if len(results) >= num_results:
                    break

        return results

    def get_query_suggestions(self, user_id: str, current_context: str = None) -> list[str]:
        """Get personalized query suggestions for a user.

        Args:
            user_id: User identifier
            current_context: Current page/context (e.g., 'sales', 'sentiment')

        Returns:
            List of suggested queries
        """
        suggestions = []

        # Get user's recent queries that worked well
        user_positive = [
            i["item_id"] for i in self._interactions
            if i["user_id"] == user_id and i.get("rating", 0) >= 4
        ]

        # Get collaborative recommendations
        recs = self.get_recommendations_local(user_id, num_results=5)

        # Context-based suggestions
        context_suggestions = {
            "sales": [
                "What are the top selling products?",
                "Show revenue by category",
                "Which region has highest sales?",
            ],
            "sentiment": [
                "What are customers complaining about?",
                "Show me positive reviews",
                "What's the average rating?",
            ],
            "forecast": [
                "Forecast next month's revenue",
                "What's the sales trend?",
                "Predict holiday season performance",
            ],
        }

        # Add context suggestions
        if current_context and current_context in context_suggestions:
            suggestions.extend(context_suggestions[current_context][:2])

        # Add collaborative recommendations
        for rec in recs[:2]:
            if rec["item_id"] not in suggestions:
                suggestions.append(rec["item_id"])

        # Add popular queries if needed
        if len(suggestions) < 5:
            popular = self._get_popular_items(5, "query")
            for p in popular:
                if p["item_id"] not in suggestions:
                    suggestions.append(p["item_id"])
                    if len(suggestions) >= 5:
                        break

        return suggestions[:5]

    # ========================================
    # AWS Personalize Methods (for production)
    # ========================================

    def get_recommendations_aws(
        self,
        campaign_arn: str,
        user_id: str,
        num_results: int = 5
    ) -> list[dict] | None:
        """Get recommendations from AWS Personalize campaign."""
        if not self._enabled:
            return None

        try:
            response = self.runtime_client.get_recommendations(
                campaignArn=campaign_arn,
                userId=user_id,
                numResults=num_results
            )

            return [
                {"item_id": item["itemId"], "score": item.get("score", 0)}
                for item in response.get("itemList", [])
            ]

        except ClientError:
            return None

    def put_events(self, tracking_id: str, user_id: str, events: list[dict]) -> bool:
        """Send events to AWS Personalize for real-time learning."""
        if not self._enabled:
            return False

        try:
            # Format events for Personalize
            formatted_events = []
            for event in events:
                formatted_events.append({
                    "eventType": event.get("type", "click"),
                    "eventValue": event.get("value", 1.0),
                    "itemId": event["item_id"],
                    "sentAt": datetime.utcnow()
                })

            self.client.put_events(
                trackingId=tracking_id,
                userId=user_id,
                sessionId=hashlib.md5(user_id.encode()).hexdigest(),
                eventList=formatted_events
            )
            return True

        except ClientError:
            return False


@lru_cache(maxsize=1)
def get_personalize_client() -> PersonalizeClient | None:
    """Get cached Personalize client."""
    try:
        return PersonalizeClient()
    except Exception:
        return None
