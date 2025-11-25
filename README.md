# 🎯 Marketing Intelligence Agent

An AI-powered marketing analyst that answers questions about sales, customer sentiment, and business trends.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-14%20passed-brightgreen.svg)]()

## 🚀 Live Demo

**[Try it on HuggingFace Spaces →](https://huggingface.co/spaces/YOUR_USERNAME/marketing-intelligence-agent)**

## 📊 What It Does

Ask natural language questions and get AI-powered insights:

- **"What products drove revenue growth?"** → Sales analysis with revenue breakdowns
- **"What are customers complaining about?"** → Sentiment analysis from 40K+ reviews
- **"Forecast next month's electronics sales"** → Time-series predictions

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INTERFACE                           │
│  Streamlit Web App (HuggingFace Spaces)                    │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTP POST /query
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                       API LAYER                             │
│  FastAPI (AWS Lambda / Local)                              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   AGENT LAYER (LangGraph)                   │
│                                                             │
│    ┌─────────────────────────────────────────────────┐     │
│    │         ORCHESTRATOR AGENT                      │     │
│    │  • Intent classification (Grok 4.1 Fast)        │     │
│    │  • Task routing                                 │     │
│    │  • Response synthesis                           │     │
│    └────┬─────────────────┬──────────────────┬───────┘     │
│         │                 │                  │             │
│         ▼                 ▼                  ▼             │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐      │
│  │   SALES     │   │  SENTIMENT  │   │  FORECAST   │      │
│  │   AGENT     │   │    AGENT    │   │   AGENT     │      │
│  │  (pandas)   │   │ (RAG+Qdrant)│   │ (time-series)│     │
│  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘      │
└─────────┼─────────────────┼─────────────────┼─────────────┘
          │                 │                 │
          ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────────┐
│                       DATA LAYER                            │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │  Parquet    │    │   Qdrant    │    │   Prophet   │     │
│  │ (100K orders)│   │ (40K chunks)│    │  (forecast) │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **LLM** | Grok 4.1 Fast (xAI) - $0.20/1M tokens |
| **Orchestration** | LangGraph state machine |
| **Vector DB** | Qdrant Cloud (hybrid search) |
| **Embeddings** | sentence-transformers (MiniLM) |
| **Backend** | FastAPI + AWS Lambda |
| **Frontend** | Streamlit |
| **Monitoring** | Langfuse |
| **Evaluation** | RAGAS |

## 📦 Dataset

[Olist Brazilian E-commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce):
- 100K+ orders (2016-2018)
- 40K+ customer reviews
- 9 interconnected tables
- ~50MB total

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/marketing-intelligence-agent
cd marketing-intelligence-agent

# Install dependencies
uv sync --all-extras

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Download data
kaggle datasets download olistbr/brazilian-ecommerce -p data/raw --unzip

# Run ETL
uv run python -m src.data.etl

# Index to Qdrant
uv run python -m src.retrieval.index

# Start API
uv run uvicorn src.api.main:app --reload

# Start UI (new terminal)
uv run streamlit run src/app/streamlit_app.py
```

## 🔑 Environment Variables

```bash
# Required
XAI_API_KEY=xai-...
QDRANT_URL=https://...cloud.qdrant.io
QDRANT_API_KEY=...

# Optional
GROQ_API_KEY=gsk_...  # For development
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
```

## 🧪 Testing

```bash
# Run all tests
uv run pytest

# With coverage
uv run pytest --cov=src --cov-report=html

# Specific test suites
uv run pytest tests/unit
uv run pytest tests/integration
uv run pytest tests/e2e  # Requires API keys
```

## 📁 Project Structure

```
├── src/
│   ├── agents/          # LangGraph agents
│   │   ├── orchestrator.py
│   │   ├── sales_agent.py
│   │   ├── sentiment_agent.py
│   │   └── forecast_agent.py
│   ├── retrieval/       # RAG pipeline
│   │   ├── rag_chain.py
│   │   └── index.py
│   ├── api/             # FastAPI backend
│   ├── app/             # Streamlit frontend
│   └── evaluation/      # RAGAS harness
├── tests/
├── data/
│   ├── raw/             # Olist CSVs
│   └── processed/       # Parquet + lexical corpus
├── infrastructure/
│   └── docker/
└── huggingface/         # HF Spaces deployment
```

## 📈 Performance

| Metric | Value |
|--------|-------|
| Avg response time | ~3-5s |
| Avg cost per query | ~$0.02 |
| Test coverage | 14 tests passing |
| RAGAS faithfulness | Target >0.85 |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Run tests: `uv run pytest`
4. Submit a PR

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

Built as an AI Engineering portfolio project demonstrating:
- Multi-agent orchestration with LangGraph
- Production RAG systems with hybrid search
- Cost-effective LLM integration
- Full-stack deployment

**[⭐ Star this repo](https://github.com/YOUR_USERNAME/marketing-intelligence-agent)** if you found it helpful!
