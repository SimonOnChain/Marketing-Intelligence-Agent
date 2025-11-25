"""Streamlit front-end for the Marketing Intelligence Agent."""

from __future__ import annotations

import os
from typing import Any, Dict

import httpx
import streamlit as st

API_URL = os.getenv("MARKETING_AGENT_API", "http://localhost:8000/query")

st.set_page_config(page_title="Marketing Intelligence Agent", page_icon="🎯", layout="wide")
st.title("🎯 Marketing Intelligence Agent")
st.caption("Sales, sentiment, and forecast insights powered by LangGraph + RAG.")


def call_api(query: str) -> Dict[str, Any]:
    with httpx.Client(timeout=30.0) as client:
        response = client.post(API_URL, json={"query": query, "include_sources": True})
        response.raise_for_status()
        return response.json()


examples = [
    "What products drove revenue growth last quarter?",
    "Summarize the top customer complaints for smart home devices.",
    "Forecast electronics revenue for next month.",
]

cols = st.columns(len(examples))
for idx, example in enumerate(examples):
    if cols[idx].button(example):
        st.session_state["query_input"] = example

query = st.text_input(
    "Ask anything about sales, sentiment, or forecasts",
    key="query_input",
    placeholder="e.g. Compare headphones vs smart home devices performance",
)

if st.button("🔍 Analyze", type="primary"):
    if not query:
        st.warning("Enter a question first.")
    else:
        with st.spinner("Crunching the numbers..."):
            try:
                result = call_api(query)
            except Exception as exc:
                st.error(f"API error: {exc}")
            else:
                st.markdown("### 💡 Answer")
                st.markdown(result["answer"])

                metrics = st.columns(3)
                metrics[0].metric("⏱️ Time", f"{result['execution_time']:.1f}s")
                metrics[1].metric("💰 Cost", f"${result['cost']:.4f}")
                agents = ", ".join(result.get("agents_used", [])) or "n/a"
                metrics[2].metric("🤖 Agents", agents)

                if result.get("sources"):
                    with st.expander("📚 Sources"):
                        for source in result["sources"]:
                            st.markdown(f"- {source.get('text', '')[:200]}...")

st.divider()
st.caption("Powered by Grok 4.1 Fast, LangGraph, and Qdrant.")

