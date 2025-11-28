"""Amazon Comprehend integration for NLP analysis."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import boto3
from botocore.exceptions import ClientError

from src.config.settings import get_settings


class ComprehendClient:
    """Client for Amazon Comprehend NLP services."""

    def __init__(self):
        """Initialize Comprehend client."""
        settings = get_settings()

        self.client = boto3.client(
            "comprehend",
            region_name=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id.get_secret_value() if settings.aws_access_key_id else None,
            aws_secret_access_key=settings.aws_secret_access_key.get_secret_value() if settings.aws_secret_access_key else None,
        )
        self._enabled = bool(settings.aws_access_key_id)

    @property
    def enabled(self) -> bool:
        """Check if Comprehend is available."""
        return self._enabled

    def analyze_sentiment(self, text: str, language: str = "en") -> dict[str, Any]:
        """Analyze sentiment of a single text.

        Args:
            text: Text to analyze (max 5000 bytes)
            language: Language code (default: en)

        Returns:
            Dict with sentiment and confidence scores
        """
        if not self._enabled:
            return self._fallback_sentiment(text)

        try:
            # Truncate if too long (Comprehend limit is 5000 bytes)
            text = text[:5000]

            response = self.client.detect_sentiment(
                Text=text,
                LanguageCode=language
            )

            return {
                "sentiment": response["Sentiment"],  # POSITIVE, NEGATIVE, NEUTRAL, MIXED
                "scores": {
                    "positive": response["SentimentScore"]["Positive"],
                    "negative": response["SentimentScore"]["Negative"],
                    "neutral": response["SentimentScore"]["Neutral"],
                    "mixed": response["SentimentScore"]["Mixed"],
                },
                "source": "comprehend",
            }

        except ClientError as e:
            # Fallback to simple analysis
            return self._fallback_sentiment(text)

    def analyze_sentiment_batch(self, texts: list[str], language: str = "en") -> list[dict[str, Any]]:
        """Analyze sentiment of multiple texts (up to 25).

        Args:
            texts: List of texts to analyze
            language: Language code

        Returns:
            List of sentiment results
        """
        if not self._enabled:
            return [self._fallback_sentiment(t) for t in texts]

        try:
            # Comprehend batch limit is 25 documents
            texts = [t[:5000] for t in texts[:25]]

            response = self.client.batch_detect_sentiment(
                TextList=texts,
                LanguageCode=language
            )

            results = []
            for item in response["ResultList"]:
                results.append({
                    "sentiment": item["Sentiment"],
                    "scores": {
                        "positive": item["SentimentScore"]["Positive"],
                        "negative": item["SentimentScore"]["Negative"],
                        "neutral": item["SentimentScore"]["Neutral"],
                        "mixed": item["SentimentScore"]["Mixed"],
                    },
                    "source": "comprehend",
                })

            return results

        except ClientError:
            return [self._fallback_sentiment(t) for t in texts]

    def extract_key_phrases(self, text: str, language: str = "en") -> list[dict[str, Any]]:
        """Extract key phrases from text.

        Args:
            text: Text to analyze
            language: Language code

        Returns:
            List of key phrases with scores
        """
        if not self._enabled:
            return []

        try:
            text = text[:5000]

            response = self.client.detect_key_phrases(
                Text=text,
                LanguageCode=language
            )

            return [
                {
                    "text": phrase["Text"],
                    "score": phrase["Score"],
                }
                for phrase in response["KeyPhrases"]
            ]

        except ClientError:
            return []

    def extract_entities(self, text: str, language: str = "en") -> list[dict[str, Any]]:
        """Extract named entities from text.

        Args:
            text: Text to analyze
            language: Language code

        Returns:
            List of entities with type and score
        """
        if not self._enabled:
            return []

        try:
            text = text[:5000]

            response = self.client.detect_entities(
                Text=text,
                LanguageCode=language
            )

            return [
                {
                    "text": entity["Text"],
                    "type": entity["Type"],  # PERSON, LOCATION, ORGANIZATION, etc.
                    "score": entity["Score"],
                }
                for entity in response["Entities"]
            ]

        except ClientError:
            return []

    def analyze_review(self, review_text: str) -> dict[str, Any]:
        """Comprehensive analysis of a customer review.

        Args:
            review_text: The review text

        Returns:
            Complete analysis with sentiment, key phrases, and entities
        """
        sentiment = self.analyze_sentiment(review_text)
        key_phrases = self.extract_key_phrases(review_text)
        entities = self.extract_entities(review_text)

        # Determine main topics from key phrases
        topics = [p["text"] for p in key_phrases[:5]]

        # Determine if it's a complaint
        is_complaint = (
            sentiment["sentiment"] == "NEGATIVE" or
            sentiment["scores"]["negative"] > 0.6
        )

        return {
            "sentiment": sentiment,
            "key_phrases": key_phrases,
            "entities": entities,
            "topics": topics,
            "is_complaint": is_complaint,
            "summary": {
                "overall": sentiment["sentiment"].lower(),
                "confidence": max(sentiment["scores"].values()),
                "main_topics": topics[:3],
            }
        }

    def _fallback_sentiment(self, text: str) -> dict[str, Any]:
        """Simple keyword-based sentiment fallback."""
        text_lower = text.lower()

        positive_words = ["good", "great", "excellent", "amazing", "love", "best", "perfect", "happy", "satisfied", "recommend"]
        negative_words = ["bad", "terrible", "awful", "hate", "worst", "poor", "disappointed", "broken", "never", "problem"]

        pos_count = sum(1 for w in positive_words if w in text_lower)
        neg_count = sum(1 for w in negative_words if w in text_lower)

        total = pos_count + neg_count + 1  # +1 to avoid division by zero

        if pos_count > neg_count:
            sentiment = "POSITIVE"
            pos_score = pos_count / total
            neg_score = neg_count / total
        elif neg_count > pos_count:
            sentiment = "NEGATIVE"
            pos_score = pos_count / total
            neg_score = neg_count / total
        else:
            sentiment = "NEUTRAL"
            pos_score = 0.3
            neg_score = 0.3

        return {
            "sentiment": sentiment,
            "scores": {
                "positive": pos_score,
                "negative": neg_score,
                "neutral": 1 - pos_score - neg_score,
                "mixed": 0.0,
            },
            "source": "fallback",
        }


@lru_cache(maxsize=1)
def get_comprehend_client() -> ComprehendClient | None:
    """Get cached Comprehend client if AWS is configured."""
    settings = get_settings()

    if not settings.aws_access_key_id:
        return None

    try:
        return ComprehendClient()
    except Exception:
        return None
