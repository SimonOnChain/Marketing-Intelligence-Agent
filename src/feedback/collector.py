"""Feedback collection and ML-based learning system."""

from __future__ import annotations

import json
import hashlib
import pickle
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np

DATA_DIR = Path(__file__).parent.parent.parent / "data"
FEEDBACK_DIR = DATA_DIR / "feedback"
FEEDBACK_FILE = FEEDBACK_DIR / "feedback_history.jsonl"
MODEL_FILE = FEEDBACK_DIR / "quality_model.pkl"
VECTORIZER_FILE = FEEDBACK_DIR / "vectorizer.pkl"


class FeedbackCollector:
    """Collects user feedback and uses ML to improve response quality scoring."""

    def __init__(self):
        self.feedback_file = FEEDBACK_FILE
        self.feedback_file.parent.mkdir(parents=True, exist_ok=True)
        self._model = None
        self._vectorizer = None
        self._feedback_cache: list[dict] = []
        self._load_feedback()
        self._load_model()  # Load persisted model if exists

    def _load_feedback(self):
        """Load existing feedback from file."""
        if self.feedback_file.exists():
            try:
                with open(self.feedback_file, "r", encoding="utf-8") as f:
                    self._feedback_cache = [json.loads(line) for line in f if line.strip()]
            except Exception:
                self._feedback_cache = []

    def _load_model(self):
        """Load persisted ML model and vectorizer."""
        try:
            if MODEL_FILE.exists() and VECTORIZER_FILE.exists():
                with open(MODEL_FILE, "rb") as f:
                    self._model = pickle.load(f)
                with open(VECTORIZER_FILE, "rb") as f:
                    self._vectorizer = pickle.load(f)
        except Exception:
            self._model = None
            self._vectorizer = None

    def _save_model(self):
        """Persist ML model and vectorizer to disk."""
        if self._model is not None and self._vectorizer is not None:
            try:
                with open(MODEL_FILE, "wb") as f:
                    pickle.dump(self._model, f)
                with open(VECTORIZER_FILE, "wb") as f:
                    pickle.dump(self._vectorizer, f)
            except Exception:
                pass  # Silent fail on save

    def _save_feedback(self, feedback: dict):
        """Append feedback to file."""
        with open(self.feedback_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(feedback) + "\n")
        self._feedback_cache.append(feedback)

    def _query_hash(self, query: str) -> str:
        """Create a hash for the query."""
        return hashlib.md5(query.lower().strip().encode()).hexdigest()[:12]

    def record_feedback(
        self,
        query: str,
        response: str,
        rating: Literal["positive", "negative"],
        intent: str | None = None,
        agents_used: list[str] | None = None,
        execution_time: float | None = None,
        comment: str | None = None,
    ) -> dict:
        """Record user feedback on a response."""
        feedback = {
            "timestamp": datetime.utcnow().isoformat(),
            "query_hash": self._query_hash(query),
            "query": query,
            "response_preview": response[:500] if response else "",
            "rating": rating,
            "rating_score": 1 if rating == "positive" else 0,
            "intent": intent,
            "agents_used": agents_used or [],
            "execution_time": execution_time,
            "comment": comment,
        }
        self._save_feedback(feedback)
        self._retrain_model()
        return feedback

    def get_feedback_stats(self) -> dict:
        """Get statistics about collected feedback."""
        if not self._feedback_cache:
            return {
                "total": 0,
                "positive": 0,
                "negative": 0,
                "positive_rate": 0.0,
                "by_intent": {},
                "by_agent": {},
                "recent_trend": "neutral",
            }

        positive = sum(1 for f in self._feedback_cache if f.get("rating") == "positive")
        negative = len(self._feedback_cache) - positive

        # Stats by intent
        by_intent = {}
        for f in self._feedback_cache:
            intent = f.get("intent", "unknown")
            if intent not in by_intent:
                by_intent[intent] = {"total": 0, "positive": 0}
            by_intent[intent]["total"] += 1
            if f.get("rating") == "positive":
                by_intent[intent]["positive"] += 1

        # Stats by agent
        by_agent = {}
        for f in self._feedback_cache:
            for agent in f.get("agents_used", []):
                if agent not in by_agent:
                    by_agent[agent] = {"total": 0, "positive": 0}
                by_agent[agent]["total"] += 1
                if f.get("rating") == "positive":
                    by_agent[agent]["positive"] += 1

        # Recent trend (last 10 vs previous 10)
        recent_trend = "neutral"
        if len(self._feedback_cache) >= 20:
            recent = self._feedback_cache[-10:]
            previous = self._feedback_cache[-20:-10]
            recent_positive = sum(1 for f in recent if f.get("rating") == "positive") / 10
            prev_positive = sum(1 for f in previous if f.get("rating") == "positive") / 10
            if recent_positive > prev_positive + 0.1:
                recent_trend = "improving"
            elif recent_positive < prev_positive - 0.1:
                recent_trend = "declining"

        return {
            "total": len(self._feedback_cache),
            "positive": positive,
            "negative": negative,
            "positive_rate": positive / len(self._feedback_cache) if self._feedback_cache else 0,
            "by_intent": by_intent,
            "by_agent": by_agent,
            "recent_trend": recent_trend,
        }

    def _retrain_model(self):
        """Retrain the ML model with new feedback data."""
        if len(self._feedback_cache) < 10:
            return  # Need minimum data

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression

            queries = [f["query"] for f in self._feedback_cache]
            labels = [f["rating_score"] for f in self._feedback_cache]

            # Check if we have both positive and negative samples
            if len(set(labels)) < 2:
                return  # Need both classes to train

            self._vectorizer = TfidfVectorizer(max_features=500, stop_words="english")
            X = self._vectorizer.fit_transform(queries)

            self._model = LogisticRegression(max_iter=200)
            self._model.fit(X, labels)

            # Persist the model
            self._save_model()

        except Exception:
            self._model = None
            self._vectorizer = None

    def predict_quality(self, query: str) -> dict:
        """Predict expected response quality for a query based on historical feedback."""
        if self._model is None or self._vectorizer is None:
            return {
                "confidence": 0.5,
                "prediction": "unknown",
                "similar_queries_positive_rate": None,
                "suggestion": None,
            }

        try:
            X = self._vectorizer.transform([query])
            proba = self._model.predict_proba(X)[0]

            # Get confidence for positive class
            positive_idx = list(self._model.classes_).index(1) if 1 in self._model.classes_ else 0
            confidence = proba[positive_idx]

            prediction = "likely_positive" if confidence > 0.6 else "likely_negative" if confidence < 0.4 else "uncertain"

            # Find similar queries
            similar_positive_rate = self._find_similar_feedback_rate(query)

            suggestion = None
            if confidence < 0.4:
                suggestion = "Consider rephrasing your question for better results."
            elif confidence < 0.6:
                suggestion = "This type of question has mixed results. Try being more specific."

            return {
                "confidence": float(confidence),
                "prediction": prediction,
                "similar_queries_positive_rate": similar_positive_rate,
                "suggestion": suggestion,
            }

        except Exception:
            return {
                "confidence": 0.5,
                "prediction": "unknown",
                "similar_queries_positive_rate": None,
                "suggestion": None,
            }

    def _find_similar_feedback_rate(self, query: str) -> float | None:
        """Find positive rate for similar queries."""
        if not self._feedback_cache or self._vectorizer is None:
            return None

        try:
            query_vec = self._vectorizer.transform([query]).toarray()[0]
            all_vecs = self._vectorizer.transform([f["query"] for f in self._feedback_cache]).toarray()

            # Cosine similarity
            similarities = np.dot(all_vecs, query_vec) / (
                np.linalg.norm(all_vecs, axis=1) * np.linalg.norm(query_vec) + 1e-8
            )

            # Get top 5 similar
            top_indices = np.argsort(similarities)[-5:]
            similar_feedback = [self._feedback_cache[i] for i in top_indices if similarities[i] > 0.3]

            if not similar_feedback:
                return None

            positive = sum(1 for f in similar_feedback if f.get("rating") == "positive")
            return positive / len(similar_feedback)

        except Exception:
            return None

    def get_improvement_suggestions(self) -> list[dict]:
        """Get suggestions for improving the system based on feedback patterns."""
        suggestions = []
        stats = self.get_feedback_stats()

        if stats["total"] < 10:
            return [{"type": "info", "message": "Need more feedback to generate suggestions."}]

        # Check overall positive rate
        if stats["positive_rate"] < 0.6:
            suggestions.append({
                "type": "warning",
                "message": f"Overall satisfaction is {stats['positive_rate']:.0%}. Consider reviewing response quality.",
            })

        # Check by intent
        for intent, data in stats["by_intent"].items():
            if data["total"] >= 5:
                rate = data["positive"] / data["total"]
                if rate < 0.5:
                    suggestions.append({
                        "type": "warning",
                        "message": f"'{intent}' queries have low satisfaction ({rate:.0%}). Review {intent} agent.",
                    })

        # Check by agent
        for agent, data in stats["by_agent"].items():
            if data["total"] >= 5:
                rate = data["positive"] / data["total"]
                if rate < 0.5:
                    suggestions.append({
                        "type": "warning",
                        "message": f"'{agent}' has low satisfaction ({rate:.0%}). Consider improvements.",
                    })

        # Check trend
        if stats["recent_trend"] == "declining":
            suggestions.append({
                "type": "alert",
                "message": "Recent feedback shows declining satisfaction. Investigate recent changes.",
            })
        elif stats["recent_trend"] == "improving":
            suggestions.append({
                "type": "positive",
                "message": "Feedback trend is improving! Recent changes are working well.",
            })

        return suggestions if suggestions else [{"type": "positive", "message": "System is performing well based on feedback."}]

    def get_recent_feedback(self, limit: int = 10) -> list[dict]:
        """Get recent feedback entries."""
        return self._feedback_cache[-limit:][::-1]

    def get_successful_queries(self, limit: int = 5) -> list[str]:
        """Get examples of queries that received positive feedback."""
        positive_queries = [
            f["query"] for f in self._feedback_cache
            if f.get("rating") == "positive"
        ]
        # Return unique queries, most recent first
        seen = set()
        unique = []
        for q in reversed(positive_queries):
            if q not in seen:
                seen.add(q)
                unique.append(q)
                if len(unique) >= limit:
                    break
        return unique

    def suggest_similar_successful_query(self, query: str) -> str | None:
        """Suggest a similar query that worked well in the past."""
        if not self._vectorizer or not self._feedback_cache:
            return None

        try:
            # Get positive feedback queries
            positive_feedback = [
                f for f in self._feedback_cache
                if f.get("rating") == "positive"
            ]

            if not positive_feedback:
                return None

            # Vectorize the input query and positive queries
            query_vec = self._vectorizer.transform([query]).toarray()[0]
            positive_queries = [f["query"] for f in positive_feedback]
            positive_vecs = self._vectorizer.transform(positive_queries).toarray()

            # Find most similar positive query
            similarities = np.dot(positive_vecs, query_vec) / (
                np.linalg.norm(positive_vecs, axis=1) * np.linalg.norm(query_vec) + 1e-8
            )

            best_idx = np.argmax(similarities)
            if similarities[best_idx] > 0.3:  # Threshold for similarity
                return positive_queries[best_idx]

            return None

        except Exception:
            return None

    def get_query_insights(self, query: str) -> dict:
        """Get comprehensive insights about a query before submitting."""
        prediction = self.predict_quality(query)
        suggestion = self.suggest_similar_successful_query(query)
        successful_examples = self.get_successful_queries(3)

        return {
            "prediction": prediction,
            "similar_successful_query": suggestion,
            "successful_examples": successful_examples,
            "model_trained": self._model is not None,
            "total_feedback": len(self._feedback_cache),
        }


@lru_cache(maxsize=1)
def get_feedback_collector() -> FeedbackCollector:
    """Get a cached feedback collector instance."""
    return FeedbackCollector()
