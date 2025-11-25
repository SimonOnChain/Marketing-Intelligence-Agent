---
title: Marketing Intelligence Agent
emoji: 🎯
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: 1.30.0
app_file: app.py
pinned: true
license: mit
---

# 🎯 Marketing Intelligence Agent

An AI-powered marketing analyst that answers questions about sales, customer sentiment, and business trends using the Olist Brazilian E-commerce dataset.

## Features

- **📊 Sales Analysis**: Revenue breakdowns, top products, category comparisons
- **💬 Sentiment Analysis**: Customer feedback themes, complaints, praise patterns  
- **🔮 Forecasting**: Trend predictions and anomaly detection
- **🔍 Hybrid RAG Search**: Combines semantic + keyword search for accurate retrieval

## Dataset

Analyzes the [Olist Brazilian E-commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) dataset:
- 100K+ orders (2016-2018)
- 40K+ customer reviews
- 70+ product categories
- 9 interconnected data tables

## Architecture

```
User Query → Intent Classification → Agent Routing
                                         ↓
                    ┌────────────────────┼────────────────────┐
                    ↓                    ↓                    ↓
              Sales Agent         Sentiment Agent       Forecast Agent
              (pandas SQL)         (RAG + Qdrant)        (time series)
                    ↓                    ↓                    ↓
                    └────────────────────┼────────────────────┘
                                         ↓
                              Response Synthesis
                                         ↓
                                   Final Answer
```

## Tech Stack

- **LLM**: Grok 4.1 Fast (xAI) - cost-effective with 2M context
- **Orchestration**: LangGraph state machine
- **Vector DB**: Qdrant Cloud (hybrid search)
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Frontend**: Streamlit
- **Monitoring**: Langfuse

## Example Questions

- "What products drove revenue growth last quarter?"
- "What are customers complaining about most?"
- "Compare electronics vs furniture sentiment"
- "Forecast next month's sales"

## Author

Built as an AI Engineering portfolio project demonstrating:
- Multi-agent orchestration
- Production RAG systems
- LLM integration best practices

---

*Powered by Grok 4.1 Fast, LangGraph, and Qdrant*

