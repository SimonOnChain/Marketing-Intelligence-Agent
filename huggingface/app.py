"""Streamlit app for HuggingFace Spaces deployment."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import streamlit as st

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Set environment variables for HF Spaces before importing modules
os.environ.setdefault("DATA_RAW_DIR", "data/raw")
os.environ.setdefault("DATA_PROCESSED_DIR", "data/processed")

from src.agents.orchestrator import Orchestrator

st.set_page_config(
    page_title="Marketing Intelligence Agent",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6B7280;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
    }
    .source-card {
        background: #F3F4F6;
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        border-left: 3px solid #667eea;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_orchestrator_instance() -> Orchestrator:
    """Initialize orchestrator once and cache it."""
    return Orchestrator()


def run_query(query: str) -> dict[str, Any]:
    """Execute query through orchestrator."""
    orchestrator = get_orchestrator_instance()
    start = time.perf_counter()
    result = orchestrator.invoke(query)
    elapsed = time.perf_counter() - start
    
    return {
        "answer": result.get("final_answer", "No answer generated."),
        "sources": result.get("sources", []),
        "intent": result.get("intent", "unknown"),
        "agents_used": result.get("agents_used", []),
        "execution_time": round(elapsed, 2),
        "cost": round(elapsed * 0.001, 4),  # Rough estimate
    }


# Header
st.markdown('<p class="main-header">🎯 Marketing Intelligence Agent</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-powered insights from 100K+ orders and customer reviews</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This AI agent analyzes the **Olist Brazilian E-commerce** dataset:
    
    - 📦 **100K+** orders
    - 💬 **40K+** customer reviews  
    - 🏷️ **70+** product categories
    - 📅 **2016-2018** timeframe
    
    ---
    
    **Powered by:**
    - 🧠 Grok 4.1 Fast (xAI)
    - 🔄 LangGraph
    - 🔍 Qdrant Vector DB
    - 📊 Hybrid RAG Search
    """)
    
    st.divider()
    
    st.markdown("**🔗 Links**")
    st.markdown("[📁 GitHub Repo](https://github.com/)")
    st.markdown("[📖 Documentation](https://github.com/)")

# Example questions
st.markdown("### 💡 Try these examples:")
examples = [
    "What products drove revenue growth?",
    "What are customers complaining about?",
    "Summarize sentiment for electronics",
    "Compare furniture vs home decor sales",
    "What's the forecast for next month?",
]

cols = st.columns(len(examples))
selected_example = None
for idx, example in enumerate(examples):
    if cols[idx].button(example, key=f"ex_{idx}", use_container_width=True):
        selected_example = example

# Query input
query = st.text_area(
    "🔍 Ask your question:",
    value=selected_example or "",
    height=100,
    placeholder="e.g., What products have the best customer reviews?",
    key="query_input",
)

# Analyze button
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    analyze_clicked = st.button("🚀 Analyze", type="primary", use_container_width=True)

# Process query
if analyze_clicked and query:
    with st.spinner("🔄 Analyzing your question..."):
        try:
            result = run_query(query)
            
            # Display metrics
            st.markdown("---")
            metric_cols = st.columns(4)
            metric_cols[0].metric("⏱️ Time", f"{result['execution_time']}s")
            metric_cols[1].metric("💰 Est. Cost", f"${result['cost']:.4f}")
            metric_cols[2].metric("🎯 Intent", result['intent'].title())
            agents = ", ".join(result['agents_used']) if result['agents_used'] else "N/A"
            metric_cols[3].metric("🤖 Agents", agents)
            
            # Display answer
            st.markdown("### 💡 Answer")
            st.markdown(result["answer"])
            
            # Display sources
            if result["sources"]:
                with st.expander(f"📚 Sources ({len(result['sources'])} reviews used)", expanded=False):
                    for i, source in enumerate(result["sources"][:10], 1):
                        text = source.get("text", "")[:300]
                        score = source.get("score", 0)
                        st.markdown(f"""
                        <div class="source-card">
                            <strong>[{i}]</strong> {text}...
                            <br><small>Relevance: {score:.2%}</small>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    if len(result["sources"]) > 10:
                        st.caption(f"+ {len(result['sources']) - 10} more sources")
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("Make sure all API keys are configured in the Space secrets.")

elif analyze_clicked and not query:
    st.warning("⚠️ Please enter a question first.")

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit, LangGraph, and Grok 4.1 Fast | [View Source](https://github.com/)")

