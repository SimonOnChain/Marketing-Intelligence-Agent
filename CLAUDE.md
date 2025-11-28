# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Marketing Intelligence Agent - an AI-powered marketing analyst that answers natural language questions about sales, customer sentiment, and business trends using the Olist Brazilian E-commerce dataset (100K+ orders, 40K+ reviews).

## Common Commands

```bash
# Install dependencies
uv sync --all-extras

# Run full setup (downloads data, runs ETL, indexes to Qdrant)
.\scripts\setup.ps1

# Start API server
uv run uvicorn src.api.main:app --reload --port 8000

# Start Streamlit UI (separate terminal)
uv run streamlit run src/app/streamlit_app.py --server.port 8500

# Run all tests
uv run pytest

# Run specific test suites
uv run pytest tests/unit -v
uv run pytest tests/integration -v
uv run pytest tests/e2e -v  # Requires API keys

# Run tests with coverage
uv run pytest --cov=src --cov-report=html

# Linting and formatting
uv run ruff check src tests
uv run ruff format src tests

# Type checking
uv run mypy src --ignore-missing-imports

# Run ETL pipeline
uv run python -m src.data.etl

# Index reviews to Qdrant
uv run python -m src.retrieval.index
```

## Architecture

### Agent Layer (LangGraph State Machine)

The core orchestration pattern is in `src/agents/orchestrator.py`:

1. **Orchestrator** receives a query and classifies intent via LLM (xAI Grok or AWS Bedrock Claude Haiku)
2. Routes to one of three specialized agents based on intent:
   - **SalesAgent** (`src/agents/sales_agent.py`) - pandas-based revenue/order aggregations
   - **SentimentAgent** (`src/agents/sentiment_agent.py`) - RAG-powered review analysis using Qdrant
   - **ForecastAgent** (`src/agents/forecast_agent.py`) - time-series predictions (moving average)
3. **Synthesizer** combines agent outputs into a cohesive response

The state machine is defined using `langgraph.graph.StateGraph` with typed state in `src/agents/state.py`.

### RAG Pipeline

`src/retrieval/rag_chain.py` implements hybrid search:
- **Vector search**: sentence-transformers MiniLM embeddings → Qdrant Cloud
- **Lexical search**: BM25-style matching against `data/processed/lexical_corpus.jsonl`
- Results are combined and passed to the SentimentAgent

### API Layer

- **FastAPI** app in `src/api/main.py` with `/query` and `/health` endpoints
- **Lambda handler** in `src/api/lambda_handler.py` (Mangum adapter) for AWS deployment
- AWS integrations in `src/aws/` (S3, Bedrock, Cognito, CloudWatch, caching)

### Configuration

All settings managed via pydantic-settings in `src/config/settings.py`. Key environment variables:
- `XAI_API_KEY` - Primary LLM (Grok 4.1 Fast)
- `QDRANT_URL`, `QDRANT_API_KEY` - Vector database
- `GROQ_API_KEY` - Alternative LLM for development
- `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` - Observability
- `BEDROCK_ENABLED`, `USE_BEDROCK_FOR_INTENT` - AWS Bedrock integration flags

### Data Flow

1. Raw Olist CSVs in `data/raw/`
2. ETL (`src/data/etl.py`) produces parquet files in `data/processed/`
3. Indexer (`src/retrieval/index.py`) creates Qdrant collection + lexical corpus JSONL

## Key Patterns

- **LLM calls** go through `src/llm/clients.py` which handles xAI/Groq API calls
- **Agent state** uses TypedDict pattern from `src/agents/state.py` for type safety
- **Settings** are cached via `@lru_cache` - use `get_settings()` to access
- **Tests** are organized by type: `tests/unit/`, `tests/integration/`, `tests/e2e/`
