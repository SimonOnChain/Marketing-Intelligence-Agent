"""Integration tests for the Orchestrator agent."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.agents.orchestrator import Orchestrator
from src.agents.state import AgentState


@pytest.fixture
def mock_llm_response():
    """Mock LLM responses for testing."""
    def _mock(messages, max_tokens=100):
        content = messages[-1]["content"].lower()
        if "classify" in messages[0]["content"].lower():
            if "revenue" in content and "sentiment" in content:
                return '{"intent": "multi", "reason": "needs sales and sentiment"}'
            elif "revenue" in content or "sales" in content:
                return '{"intent": "sales", "reason": "revenue question"}'
            elif "review" in content or "complaint" in content:
                return '{"intent": "sentiment", "reason": "sentiment question"}'
            elif "forecast" in content or "predict" in content:
                return '{"intent": "forecast", "reason": "forecast question"}'
            return '{"intent": "sales", "reason": "default"}'
        return "This is a synthesized response based on the data provided."
    return _mock


@pytest.fixture
def orchestrator(mock_llm_response):
    """Create orchestrator with mocked dependencies."""
    with patch("src.agents.orchestrator.call_xai_chat", side_effect=mock_llm_response), \
         patch("src.llm.clients.call_xai_chat", side_effect=mock_llm_response), \
         patch("src.agents.sentiment_agent.build_rag_chain") as mock_rag:
        
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {
            "answer": "Sentiment analysis complete.",
            "sources": [{"id": "1", "text": "Great product!", "score": 0.9}],
        }
        mock_rag.return_value = mock_chain
        
        yield Orchestrator()


def test_orchestrator_routes_sales_query(orchestrator, mock_llm_response):
    """Test that sales queries route to sales agent."""
    with patch("src.agents.orchestrator.call_xai_chat", side_effect=mock_llm_response):
        result = orchestrator.invoke("What is our total revenue?")
    
    assert result["intent"] == "sales"
    assert result["final_answer"] is not None


def test_orchestrator_routes_sentiment_query(orchestrator, mock_llm_response):
    """Test that sentiment queries route to sentiment agent."""
    with patch("src.agents.orchestrator.call_xai_chat", side_effect=mock_llm_response):
        result = orchestrator.invoke("What are customers complaining about in reviews?")
    
    assert result["intent"] == "sentiment"
    assert result["final_answer"] is not None


def test_orchestrator_routes_multi_query(orchestrator, mock_llm_response):
    """Test that multi-intent queries route to both agents."""
    with patch("src.agents.orchestrator.call_xai_chat", side_effect=mock_llm_response):
        result = orchestrator.invoke("What products drove revenue and what's the sentiment?")
    
    assert result["intent"] == "multi"
    assert result["final_answer"] is not None


def test_orchestrator_handles_invalid_json(orchestrator):
    """Test graceful handling of malformed LLM response."""
    def bad_response(messages, max_tokens=100):
        if "classify" in messages[0]["content"].lower():
            return "not valid json"
        return "Synthesized response."
    
    with patch("src.agents.orchestrator.call_xai_chat", side_effect=bad_response):
        result = orchestrator.invoke("Some query")
    
    # Should default to "sales" intent
    assert result["intent"] == "sales"


def test_orchestrator_collects_sources(orchestrator, mock_llm_response):
    """Test that sources are collected from sentiment agent."""
    with patch("src.agents.orchestrator.call_xai_chat", side_effect=mock_llm_response):
        result = orchestrator.invoke("What do reviews say?")
    
    assert "sources" in result

