# Marketing Intelligence Agent

A production-ready AI-powered marketing analyst that answers natural language questions about sales performance, customer sentiment, and business forecasting using a multi-agent architecture.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Deployed on AWS](https://img.shields.io/badge/AWS-EC2-orange.svg)](http://3.121.239.209:8501)

## Live Demo

**[http://3.121.239.209:8501](http://3.121.239.209:8501)** - Deployed on AWS EC2

## Overview

This project demonstrates a complete AI engineering solution combining:

- **Multi-Agent Orchestration** using LangGraph state machines
- **Retrieval-Augmented Generation (RAG)** with hybrid search (vector + lexical)
- **ML Time-Series Forecasting** using Facebook Prophet
- **Production Deployment** on AWS EC2

### Example Queries

| Query Type | Example | Agent |
|------------|---------|-------|
| Sales Analysis | *"What were the top-selling categories last month?"* | Sales Agent |
| Sentiment Analysis | *"What do customers say about delivery times?"* | Sentiment Agent (RAG) |
| Forecasting | *"How will revenue develop over the next 8 weeks?"* | Forecast Agent (Prophet) |

## Architecture

```
                              ┌─────────────────────┐
                              │    User Interface   │
                              │     (Streamlit)     │
                              └──────────┬──────────┘
                                         │
                              ┌──────────▼──────────┐
                              │      FastAPI        │
                              │    REST Endpoint    │
                              └──────────┬──────────┘
                                         │
┌────────────────────────────────────────▼────────────────────────────────────────┐
│                          LangGraph State Machine                                │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         ORCHESTRATOR                                     │   │
│  │              Intent Classification (xAI Grok / AWS Bedrock)             │   │
│  └───────┬─────────────────────┬─────────────────────┬─────────────────────┘   │
│          │                     │                     │                         │
│          ▼                     ▼                     ▼                         │
│  ┌───────────────┐     ┌───────────────┐     ┌───────────────┐                │
│  │  Sales Agent  │     │Sentiment Agent│     │Forecast Agent │                │
│  │   (Pandas)    │     │  (RAG+Qdrant) │     │   (Prophet)   │                │
│  └───────┬───────┘     └───────┬───────┘     └───────┬───────┘                │
│          │                     │                     │                         │
│          └─────────────────────┼─────────────────────┘                         │
│                                ▼                                               │
│                        ┌───────────────┐                                       │
│                        │  Synthesizer  │                                       │
│                        └───────────────┘                                       │
└────────────────────────────────────────────────────────────────────────────────┘
                                         │
          ┌──────────────────────────────┼──────────────────────────────┐
          │                              │                              │
          ▼                              ▼                              ▼
┌─────────────────┐           ┌─────────────────┐           ┌─────────────────┐
│  Parquet Files  │           │  Qdrant Cloud   │           │ Prophet Models  │
│  (100K+ Orders) │           │ (40K+ Reviews)  │           │  (Forecasting)  │
└─────────────────┘           └─────────────────┘           └─────────────────┘
```

## Key Technologies

### Multi-Agent System (LangGraph)

The orchestrator uses a state machine pattern to route queries to specialized agents:

```python
class AgentState(TypedDict):
    query: str
    intent: str
    agent_outputs: Dict[str, Any]
    final_response: str

graph = StateGraph(AgentState)
graph.add_node("classify", classify_intent)
graph.add_node("sales", sales_agent)
graph.add_node("sentiment", sentiment_agent)
graph.add_node("forecast", forecast_agent)
graph.add_conditional_edges("classify", route_to_agent)
```

### RAG Pipeline with Hybrid Search

Combines semantic vector search with lexical BM25 matching:

- **Embedding Model**: sentence-transformers/all-MiniLM-L6-v2
- **Vector Database**: Qdrant Cloud (40K+ indexed review chunks)
- **Fusion Strategy**: Reciprocal Rank Fusion (RRF)

### Prophet ML Forecasting

Time-series predictions with automatic seasonality detection:

```python
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True
)
model.fit(df)
forecast = model.predict(future_dates)
```

## Tech Stack

| Layer | Technology |
|-------|------------|
| **LLM Providers** | xAI Grok, Groq (Llama 3), AWS Bedrock (Claude) |
| **Orchestration** | LangGraph, LangChain |
| **Vector Database** | Qdrant Cloud |
| **Embeddings** | sentence-transformers (MiniLM) |
| **ML Forecasting** | Facebook Prophet |
| **Backend** | FastAPI, Uvicorn |
| **Frontend** | Streamlit |
| **Monitoring** | Langfuse (Tracing & Analytics) |
| **Infrastructure** | AWS EC2, S3 |
| **Data Processing** | Pandas, Polars, Parquet |

## Dataset

[Olist Brazilian E-commerce Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce):

- 100,000+ orders (2016-2018)
- 40,000+ customer reviews
- 9 interconnected tables
- Revenue, product categories, geolocation data

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- API keys for xAI/Groq and Qdrant

### Installation

```bash
# Clone repository
git clone https://github.com/SimonOnChain/Marketing-Intelligence-Agent.git
cd Marketing-Intelligence-Agent

# Install dependencies
uv sync --all-extras

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Download and process data
kaggle datasets download olistbr/brazilian-ecommerce -p data/raw --unzip
uv run python -m src.data.etl

# Index reviews to Qdrant
uv run python -m src.retrieval.index

# Start API server
uv run uvicorn src.api.main:app --reload --port 8000

# Start Streamlit UI (separate terminal)
uv run streamlit run src/app/streamlit_app.py --server.port 8501
```

### Environment Variables

```bash
# LLM Providers (at least one required)
XAI_API_KEY=xai-...
GROQ_API_KEY=gsk_...

# Vector Database (required)
QDRANT_URL=https://...cloud.qdrant.io
QDRANT_API_KEY=...

# Observability (optional)
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com

# AWS Integration (optional)
BEDROCK_ENABLED=false
AWS_REGION=eu-central-1
```

## Project Structure

```
Marketing-Intelligence-Agent/
├── src/
│   ├── agents/              # LangGraph multi-agent system
│   │   ├── orchestrator.py  # State machine & routing
│   │   ├── sales_agent.py   # Pandas-based analytics
│   │   ├── sentiment_agent.py # RAG-powered analysis
│   │   ├── forecast_agent.py  # Prophet ML predictions
│   │   └── state.py         # TypedDict state definitions
│   ├── retrieval/           # RAG pipeline
│   │   ├── rag_chain.py     # Hybrid search implementation
│   │   └── index.py         # Qdrant indexing
│   ├── api/                 # FastAPI backend
│   │   ├── main.py          # REST endpoints
│   │   └── lambda_handler.py # AWS Lambda adapter
│   ├── app/                 # Streamlit frontend
│   │   └── streamlit_app.py
│   ├── llm/                 # LLM client abstractions
│   ├── config/              # Pydantic settings
│   └── data/                # ETL pipeline
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── data/
│   ├── raw/                 # Olist CSV files
│   └── processed/           # Parquet + lexical corpus
├── scripts/                 # Utility scripts
└── terraform/               # Infrastructure as Code
```

## Testing

```bash
# Run all tests
uv run pytest

# Run with coverage report
uv run pytest --cov=src --cov-report=html

# Run specific test suites
uv run pytest tests/unit -v
uv run pytest tests/integration -v
uv run pytest tests/e2e -v  # Requires API keys
```

## Performance Metrics

| Metric | Value |
|--------|-------|
| Intent Classification Accuracy | >95% |
| Average Response Time | 3-5 seconds |
| RAG Retrieval Precision | ~85% |
| Prophet MAPE | ~12% |

## Author

**Simon Jokani**
Data Science, Machine Learning & AI

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*This project demonstrates production-grade AI engineering practices including multi-agent orchestration, RAG systems, ML forecasting, and cloud deployment.*
