"""AWS Bedrock integration for Claude and other foundation models."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import boto3
from botocore.exceptions import ClientError

from src.config.settings import get_settings


class BedrockClient:
    """Client for AWS Bedrock foundation models including Claude."""

    # Model IDs
    CLAUDE_SONNET = "anthropic.claude-3-sonnet-20240229-v1:0"
    CLAUDE_HAIKU = "anthropic.claude-3-haiku-20240307-v1:0"  # Fast, cheap for classification
    CLAUDE_OPUS = "anthropic.claude-3-opus-20240229-v1:0"
    TITAN_EMBED = "amazon.titan-embed-text-v1"

    def __init__(self):
        """Initialize Bedrock client."""
        settings = get_settings()

        self.client = boto3.client(
            "bedrock-runtime",
            region_name=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id.get_secret_value() if settings.aws_access_key_id else None,
            aws_secret_access_key=settings.aws_secret_access_key.get_secret_value() if settings.aws_secret_access_key else None,
        )
        self._enabled = bool(settings.aws_access_key_id)

    @property
    def enabled(self) -> bool:
        """Check if Bedrock is available."""
        return self._enabled

    def invoke_claude(
        self,
        messages: list[dict[str, str]],
        model_id: str = CLAUDE_HAIKU,
        max_tokens: int = 500,
        temperature: float = 0.0,
        system: str | None = None,
    ) -> str:
        """Invoke Claude model on Bedrock.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model_id: Bedrock model ID
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            system: Optional system prompt

        Returns:
            Model response text
        """
        if not self._enabled:
            raise RuntimeError("Bedrock not configured. Set AWS credentials.")

        # Format for Claude Messages API
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }

        if system:
            body["system"] = system

        try:
            response = self.client.invoke_model(
                modelId=model_id,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json",
            )

            response_body = json.loads(response["body"].read())
            return response_body["content"][0]["text"]

        except ClientError as e:
            raise RuntimeError(f"Bedrock invocation failed: {e}") from e

    def classify_intent_fast(self, query: str) -> str:
        """Fast intent classification using Claude Haiku.

        This is ~3x faster and ~10x cheaper than using the full reasoning model.

        Args:
            query: User's question

        Returns:
            Intent string: 'sales', 'sentiment', 'forecast', or 'multi'
        """
        system = """You classify marketing questions into EXACTLY one category.
Respond with ONLY the category name, nothing else.

Categories:
- sales: revenue, products, orders, categories, growth, performance
- sentiment: reviews, complaints, feedback, ratings, opinions
- forecast: predictions, next month, future, trends
- multi: needs both sales AND sentiment data"""

        messages = [{"role": "user", "content": query}]

        try:
            response = self.invoke_claude(
                messages=messages,
                model_id=self.CLAUDE_HAIKU,
                max_tokens=20,
                temperature=0.0,
                system=system,
            )

            # Parse response
            intent = response.strip().lower()
            valid_intents = {"sales", "sentiment", "forecast", "multi"}

            if intent in valid_intents:
                return intent

            # Fallback to keyword matching
            return self._keyword_fallback(query)

        except Exception:
            return self._keyword_fallback(query)

    def _keyword_fallback(self, query: str) -> str:
        """Fallback intent classification using keywords."""
        query_lower = query.lower()

        forecast_keywords = ["forecast", "predict", "next month", "future", "projection", "trend"]
        if any(kw in query_lower for kw in forecast_keywords):
            return "forecast"

        sentiment_keywords = ["complaint", "review", "feedback", "rating", "sentiment", "opinion"]
        if any(kw in query_lower for kw in sentiment_keywords):
            return "sentiment"

        has_sales = any(kw in query_lower for kw in ["revenue", "sales", "order", "product"])
        has_sentiment = any(kw in query_lower for kw in sentiment_keywords)

        if has_sales and has_sentiment:
            return "multi"

        return "sales"

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using Amazon Titan.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not self._enabled:
            raise RuntimeError("Bedrock not configured.")

        embeddings = []
        for text in texts:
            body = {"inputText": text}

            try:
                response = self.client.invoke_model(
                    modelId=self.TITAN_EMBED,
                    body=json.dumps(body),
                    contentType="application/json",
                    accept="application/json",
                )
                response_body = json.loads(response["body"].read())
                embeddings.append(response_body["embedding"])
            except ClientError as e:
                raise RuntimeError(f"Embedding generation failed: {e}") from e

        return embeddings

    def synthesize_response(
        self,
        question: str,
        context: dict[str, Any],
        max_tokens: int = 800,
    ) -> str:
        """Generate a synthesized response using Claude Sonnet.

        Args:
            question: Original user question
            context: Dict containing sales_result, sentiment_result, forecast_result
            max_tokens: Maximum response length

        Returns:
            Synthesized answer
        """
        system = """You are a marketing executive consultant.
Analyze the provided data and give actionable recommendations.
Be concise and focus on insights that drive business decisions."""

        prompt = f"""Question: {question}

Data:
{json.dumps(context, indent=2)}

Provide a crisp analysis with recommendations."""

        messages = [{"role": "user", "content": prompt}]

        return self.invoke_claude(
            messages=messages,
            model_id=self.CLAUDE_SONNET,
            max_tokens=max_tokens,
            temperature=0.3,
            system=system,
        )


@lru_cache(maxsize=1)
def get_bedrock_client() -> BedrockClient | None:
    """Get cached Bedrock client if AWS is configured."""
    settings = get_settings()

    if not settings.aws_access_key_id:
        return None

    try:
        client = BedrockClient()
        return client
    except Exception:
        return None
