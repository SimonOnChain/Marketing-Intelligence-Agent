"""Integration tests for the FastAPI endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_orchestrator():
    """Mock orchestrator for API tests."""
    mock = MagicMock()
    mock.invoke.return_value = {
        "intent": "sales",
        "final_answer": "Revenue increased by 15% last quarter.",
        "sources": [{"id": "1", "text": "Sample source", "score": 0.95}],
        "sales_result": {"total_revenue": 150000},
        "sentiment_result": None,
        "forecast_result": None,
        "agents_used": ["sales"],
        "input_tokens": 500,
        "output_tokens": 200,
    }
    return mock


def test_health_endpoint(client):
    """Test health check returns OK."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_query_endpoint_success(client, mock_orchestrator):
    """Test successful query execution."""
    with patch("src.api.main.get_orchestrator", return_value=mock_orchestrator):
        response = client.post(
            "/query",
            json={"query": "What is our revenue?", "include_sources": True}
        )
    
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "execution_time" in data
    assert "cost" in data
    assert len(data["answer"]) > 0  # Has some response


def test_query_endpoint_without_sources(client, mock_orchestrator):
    """Test query with sources disabled."""
    with patch("src.api.main.get_orchestrator", return_value=mock_orchestrator):
        response = client.post(
            "/query",
            json={"query": "What is our revenue?", "include_sources": False}
        )
    
    assert response.status_code == 200
    data = response.json()
    assert data["sources"] is None


def test_query_endpoint_validation_error(client):
    """Test validation rejects short queries."""
    response = client.post(
        "/query",
        json={"query": "Hi"}  # Too short (min 5 chars)
    )
    assert response.status_code == 422


def test_query_endpoint_missing_query(client):
    """Test validation rejects missing query."""
    response = client.post("/query", json={})
    assert response.status_code == 422


def test_query_method_not_allowed(client):
    """Test GET on /query returns 405."""
    response = client.get("/query")
    assert response.status_code == 405


def test_cost_estimation():
    """Test cost estimation function."""
    from src.api.main import estimate_cost
    
    result = {
        "input_tokens": 1_000_000,
        "output_tokens": 1_000_000,
    }
    cost = estimate_cost(result)
    # $0.20/1M input + $0.50/1M output = $0.70
    assert cost == 0.70


def test_cost_estimation_zero_tokens():
    """Test cost estimation with no tokens."""
    from src.api.main import estimate_cost
    
    result = {}
    cost = estimate_cost(result)
    assert cost == 0.0

