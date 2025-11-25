"""FastAPI surface for the Marketing Intelligence Agent."""

from __future__ import annotations

import time
from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.agents.orchestrator import Orchestrator

app = FastAPI(
    title="Marketing Intelligence Agent",
    description="Sales + Sentiment + Forecast agentic API",
    version="0.1.0",
)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=5, max_length=1000)
    include_sources: bool = True


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict] | None
    agents_used: list[str]
    execution_time: float
    cost: float


@lru_cache(maxsize=1)
def get_orchestrator() -> Orchestrator:
    return Orchestrator()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def run_query(payload: QueryRequest, orchestrator: Orchestrator = Depends(get_orchestrator)) -> QueryResponse:
    start = time.perf_counter()
    try:
        result = orchestrator.invoke(payload.query)
    except Exception as exc:  # pragma: no cover - FastAPI will serialize
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    elapsed = time.perf_counter() - start
    response = QueryResponse(
        answer=result["final_answer"],
        sources=result["sources"] if payload.include_sources else None,
        agents_used=result.get("agents_used", []),
        execution_time=round(elapsed, 2),
        cost=estimate_cost(result),
    )
    return response


def estimate_cost(result: dict) -> float:
    input_tokens = result.get("input_tokens", 0)
    output_tokens = result.get("output_tokens", 0)
    cost = (input_tokens / 1_000_000 * 0.20) + (output_tokens / 1_000_000 * 0.50)
    return round(cost, 4)

