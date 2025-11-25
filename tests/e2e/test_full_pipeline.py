"""End-to-end tests for the full pipeline."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


# Skip E2E tests if no API keys are configured
SKIP_E2E = not os.getenv("XAI_API_KEY")


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.mark.skipif(SKIP_E2E, reason="XAI_API_KEY not set")
class TestFullPipeline:
    """E2E tests that hit real services (requires API keys)."""

    def test_sales_query_e2e(self, client):
        """Test a real sales query through the full pipeline."""
        response = client.post(
            "/query",
            json={
                "query": "What are the top 3 product categories by revenue?",
                "include_sources": True,
            },
            timeout=30.0,
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "answer" in data
        assert len(data["answer"]) > 50  # Substantial response
        assert data["execution_time"] < 30.0
        assert data["cost"] < 1.0  # Under $1

    def test_sentiment_query_e2e(self, client):
        """Test a real sentiment query through the full pipeline."""
        response = client.post(
            "/query",
            json={
                "query": "What do customers complain about most in reviews?",
                "include_sources": True,
            },
            timeout=30.0,
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "answer" in data
        assert len(data["answer"]) > 50
        assert data["sources"] is not None
        assert len(data["sources"]) > 0

    def test_multi_intent_query_e2e(self, client):
        """Test a multi-intent query through the full pipeline."""
        response = client.post(
            "/query",
            json={
                "query": "What products drove revenue and what's the customer sentiment?",
                "include_sources": True,
            },
            timeout=45.0,
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "answer" in data
        assert len(data["answer"]) > 100  # More comprehensive response


class TestPipelineWithMocks:
    """E2E-style tests with mocked external services."""

    def test_query_flow_mocked(self, client):
        """Test the query flow with mocked LLM."""
        mock_response = {
            "intent": "sales",
            "final_answer": "Based on analysis, electronics is the top category.",
            "sources": [],
            "agents_used": ["sales"],
        }
        
        with patch("src.api.main.get_orchestrator") as mock_orch:
            mock_orch.return_value.invoke.return_value = mock_response
            
            response = client.post(
                "/query",
                json={"query": "What is the top category?"}
            )
        
        assert response.status_code == 200
        assert "electronics" in response.json()["answer"].lower()

    def test_error_handling(self, client):
        """Test graceful error handling."""
        with patch("src.api.main.get_orchestrator") as mock_orch:
            mock_orch.return_value.invoke.side_effect = Exception("LLM timeout")
            
            response = client.post(
                "/query",
                json={"query": "This will fail"}
            )
        
        assert response.status_code == 500
        assert "LLM timeout" in response.json()["detail"]

