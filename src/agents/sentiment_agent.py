"""Sentiment analysis agent built on the RAG chain with AWS Comprehend enhancement."""

from dataclasses import dataclass, field
from typing import Any

from src.agents.state import AgentState
from src.retrieval.rag_chain import RAGChain, build_rag_chain
from src.aws.comprehend import get_comprehend_client


@dataclass
class SentimentAgent:
    """Sentiment analysis agent using RAG + optional AWS Comprehend."""

    chain: RAGChain | None = None
    use_comprehend: bool = True

    def __post_init__(self) -> None:
        self.chain = self.chain or build_rag_chain()
        self._comprehend = get_comprehend_client() if self.use_comprehend else None

    def _analyze_with_comprehend(self, reviews: list[str]) -> dict[str, Any]:
        """Analyze reviews using AWS Comprehend for enhanced sentiment."""
        if not self._comprehend or not reviews:
            return {}

        # Batch analyze sentiments (Comprehend limit is 25)
        batch_size = 25
        all_results = []

        for i in range(0, len(reviews), batch_size):
            batch = reviews[i:i + batch_size]
            results = self._comprehend.analyze_sentiment_batch(batch)
            all_results.extend(results)

        # Aggregate results
        sentiment_counts = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0, "MIXED": 0}
        total_scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0, "mixed": 0.0}

        for result in all_results:
            sentiment = result.get("sentiment", "NEUTRAL")
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
            for key in total_scores:
                total_scores[key] += result.get("scores", {}).get(key, 0)

        n = len(all_results) if all_results else 1
        avg_scores = {k: v / n for k, v in total_scores.items()}

        # Extract key phrases from negative reviews
        negative_reviews = [
            reviews[i] for i, r in enumerate(all_results)
            if r.get("sentiment") == "NEGATIVE" and i < len(reviews)
        ]

        key_issues = []
        if negative_reviews and self._comprehend:
            # Sample up to 5 negative reviews for key phrase extraction
            for review in negative_reviews[:5]:
                phrases = self._comprehend.extract_key_phrases(review)
                key_issues.extend([p["text"] for p in phrases[:3]])

        # Deduplicate key issues
        key_issues = list(dict.fromkeys(key_issues))[:10]

        return {
            "comprehend_analysis": {
                "total_analyzed": len(all_results),
                "sentiment_distribution": sentiment_counts,
                "average_scores": avg_scores,
                "dominant_sentiment": max(sentiment_counts, key=sentiment_counts.get),
                "key_issues": key_issues,
                "source": all_results[0].get("source", "unknown") if all_results else "none",
            }
        }

    def invoke(self, state: AgentState) -> AgentState:
        query = state["query"]

        # Single RAG call to find relevant reviews AND get summary
        # This avoids the double RAG call that was causing latency + cost issues
        prompt = (
            f"Analyze customer reviews relevant to: {query}. "
            "Provide a summary of the sentiment themes, highlighting positive and negative feedback."
        )
        rag_result = self.chain.invoke(prompt)

        # Extract reviews from RAG source documents for Comprehend analysis
        reviews_text = []
        if rag_result and "source_documents" in rag_result:
            for doc in rag_result.get("source_documents", []):
                content = doc.page_content if hasattr(doc, "page_content") else str(doc)
                if content and len(content) > 10:
                    reviews_text.append(content[:5000])  # Comprehend limit

        # If we have reviews, enhance with Comprehend
        comprehend_result = {}
        if reviews_text:
            comprehend_result = self._analyze_with_comprehend(reviews_text)

        # Combine results (using single RAG result)
        enhanced_result = {
            "rag_summary": rag_result,
            "reviews_analyzed": len(reviews_text),
            **comprehend_result,
        }

        # If Comprehend provided insights, add them to the summary
        if comprehend_result.get("comprehend_analysis"):
            analysis = comprehend_result["comprehend_analysis"]
            enhanced_result["enhanced_summary"] = {
                "overall_sentiment": analysis["dominant_sentiment"].lower(),
                "confidence": analysis["average_scores"].get(analysis["dominant_sentiment"].lower(), 0),
                "positive_ratio": analysis["sentiment_distribution"]["POSITIVE"] / max(analysis["total_analyzed"], 1),
                "negative_ratio": analysis["sentiment_distribution"]["NEGATIVE"] / max(analysis["total_analyzed"], 1),
                "key_concerns": analysis["key_issues"],
            }

        state["sentiment_result"] = enhanced_result
        agents_used = state.get("agents_used", [])
        agents_used.append("sentiment")
        state["agents_used"] = agents_used
        return state

