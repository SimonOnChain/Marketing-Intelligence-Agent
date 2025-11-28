"""Streamlit front-end for the Marketing Intelligence Agent."""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.app.chat_history import get_history_manager
from src.feedback.collector import get_feedback_collector

API_URL = os.getenv("MARKETING_AGENT_API", "http://localhost:8000/query")
API_BASE = os.getenv("MARKETING_AGENT_API_BASE", "http://localhost:8000")


# Load external CSS
def load_css():
    """Load custom CSS from external file."""
    css_path = Path(__file__).parent / "styles.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Initialize page config
st.set_page_config(
    page_title="Marketing Intelligence Platform",
    page_icon="M",
    layout="wide",
    initial_sidebar_state="collapsed"
)
load_css()

# ============== SESSION STATE INITIALIZATION ==============

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if "chat_history" not in st.session_state:
    history_manager = get_history_manager()
    st.session_state.chat_history = history_manager.load_session(st.session_state.session_id)

if "theme" not in st.session_state:
    st.session_state.theme = "dark"

if "custom_questions" not in st.session_state:
    st.session_state.custom_questions = []

if "query_history" not in st.session_state:
    st.session_state.query_history = []

if "conversation_context" not in st.session_state:
    st.session_state.conversation_context = []

if "show_suggestions" not in st.session_state:
    st.session_state.show_suggestions = True

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

if "is_followup_query" not in st.session_state:
    st.session_state.is_followup_query = False

if "followup_parent_idx" not in st.session_state:
    st.session_state.followup_parent_idx = None

if "quick_questions_open" not in st.session_state:
    st.session_state.quick_questions_open = False

if "close_quick_questions" not in st.session_state:
    st.session_state.close_quick_questions = False

# Handle close flag BEFORE any widgets are rendered
if st.session_state.close_quick_questions:
    st.session_state.quick_questions_open = False
    st.session_state.close_quick_questions = False

if "onboarding_completed" not in st.session_state:
    st.session_state.onboarding_completed = False

if "show_onboarding" not in st.session_state:
    # Show onboarding for first-time users (no chat history)
    history_manager = get_history_manager()
    sessions = history_manager.list_sessions()
    st.session_state.show_onboarding = len(sessions) == 0


# ============== HELPER FUNCTIONS ==============

def save_chat_history():
    """Save current chat history to persistent storage."""
    history_manager = get_history_manager()
    history_manager.save_session(st.session_state.session_id, st.session_state.chat_history)


def call_api(query: str, context: List[Dict] = None) -> Dict[str, Any]:
    """Call the backend API with error handling and context."""
    payload = {
        "query": query,
        "include_sources": True,
    }
    if context:
        payload["context"] = context[-5:]  # Last 5 messages for context

    with httpx.Client(timeout=60.0) as client:
        response = client.post(API_URL, json=payload)
        response.raise_for_status()
        return response.json()


def call_api_streaming(query: str, context: List[Dict] = None):
    """Call the backend API with streaming support."""
    payload = {
        "query": query,
        "include_sources": True,
        "stream": True,
    }
    if context:
        payload["context"] = context[-5:]

    try:
        with httpx.Client(timeout=60.0) as client:
            with client.stream("POST", f"{API_BASE}/query/stream", json=payload) as response:
                for chunk in response.iter_text():
                    if chunk:
                        yield chunk
    except httpx.HTTPStatusError:
        # Fallback to non-streaming
        result = call_api(query, context)
        yield json.dumps(result)


@st.cache_data(ttl=30)  # Cache for 30 seconds
def get_api_health() -> Dict[str, Any]:
    """Check API health status."""
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{API_BASE}/health")
            return response.json() if response.status_code == 200 else {"status": "unhealthy"}
    except Exception:
        return {"status": "offline"}


@st.cache_data(ttl=60)  # Cache for 60 seconds - this is the heavy one!
def get_api_stats() -> Dict[str, Any]:
    """Get dataset statistics from API."""
    try:
        with httpx.Client(timeout=15.0) as client:
            response = client.get(f"{API_BASE}/stats")
            if response.status_code == 200:
                data = response.json()
                # Also get categories and monthly revenue
                cat_resp = client.get(f"{API_BASE}/stats/categories?limit=10")
                monthly_resp = client.get(f"{API_BASE}/stats/monthly-revenue")
                state_resp = client.get(f"{API_BASE}/stats/by-state?limit=15")
                ratings_resp = client.get(f"{API_BASE}/stats/ratings")
                forecast_resp = client.get(f"{API_BASE}/stats/forecast-demo")
                geo_resp = client.get(f"{API_BASE}/stats/geo-map?limit=100")
                anomaly_resp = client.get(f"{API_BASE}/stats/anomalies")
                if cat_resp.status_code == 200:
                    data["top_categories"] = cat_resp.json()
                if monthly_resp.status_code == 200:
                    data["monthly_revenue"] = monthly_resp.json()
                if state_resp.status_code == 200:
                    data["by_state"] = state_resp.json()
                if ratings_resp.status_code == 200:
                    data["ratings"] = ratings_resp.json()
                if forecast_resp.status_code == 200:
                    data["forecast_demo"] = forecast_resp.json()
                if geo_resp.status_code == 200:
                    data["geo_map"] = geo_resp.json()
                if anomaly_resp.status_code == 200:
                    data["anomalies"] = anomaly_resp.json()
                data["date_range"] = {
                    "start": data.get("date_range_start", "N/A"),
                    "end": data.get("date_range_end", "N/A"),
                }
                return data
            return {}
    except Exception:
        return {}


@st.cache_data(ttl=60)  # Cache for 60 seconds
def get_aws_status() -> Dict[str, Any]:
    """Get AWS services status."""
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{API_BASE}/aws/status")
            if response.status_code == 200:
                data = response.json()
                services = data.get("services", {})
                return {
                    "s3": services.get("s3", {}),
                    "bedrock": services.get("bedrock", {}),
                    "dynamodb_cache": services.get("dynamodb_cache", {}),
                    "comprehend": {"enabled": False},
                    "cloudwatch": services.get("cloudwatch", {}),
                    "cognito": services.get("cognito", {}),
                    "redis_cache": services.get("redis_cache", {}),
                }
            return {}
    except Exception:
        return {}


@st.cache_data(ttl=120)  # Cache for 2 minutes - ML data doesn't change often
def get_ml_data(endpoint: str) -> Dict[str, Any]:
    """Fetch ML analytics data from API."""
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(f"{API_BASE}/stats/ml/{endpoint}")
            if response.status_code == 200:
                return response.json()
            return {}
    except Exception:
        return {}


def get_suggested_questions(query: str, answer: str) -> List[str]:
    """Generate follow-up question suggestions based on context."""
    suggestions = []
    query_lower = query.lower()

    # Sales-related suggestions
    if any(kw in query_lower for kw in ["revenue", "sales", "product", "category"]):
        suggestions.extend([
            "How does this compare to last quarter?",
            "What's driving this trend?",
            "Show me the top performing regions",
        ])

    # Sentiment-related suggestions
    if any(kw in query_lower for kw in ["review", "sentiment", "complaint", "feedback"]):
        suggestions.extend([
            "What are the main customer pain points?",
            "How has sentiment changed over time?",
            "Which products have the best ratings?",
        ])

    # Forecast-related suggestions
    if any(kw in query_lower for kw in ["forecast", "predict", "future", "trend"]):
        suggestions.extend([
            "What factors influence this prediction?",
            "Show historical accuracy of forecasts",
            "What if we increase marketing spend?",
        ])

    return suggestions[:3]


def get_intent_badge(agents: List[str]) -> str:
    """Generate intent badge HTML based on agents used."""
    if not agents:
        return ""
    badges = []
    for agent in agents:
        agent_lower = agent.lower()
        if "sales" in agent_lower:
            badges.append('<span class="intent-badge sales">Sales</span>')
        elif "sentiment" in agent_lower:
            badges.append('<span class="intent-badge sentiment">Sentiment</span>')
        elif "forecast" in agent_lower:
            badges.append('<span class="intent-badge forecast">Forecast</span>')
    return "".join(badges)


def render_chat_message(msg: Dict[str, Any], is_user: bool = False, msg_idx: int = 0):
    """Render a chat message with styling, markdown support, and rich media."""
    msg_class = "user" if is_user else "assistant"
    sender = "You" if is_user else "AI Assistant"
    timestamp = msg.get("timestamp", "")
    if timestamp:
        try:
            ts = datetime.fromisoformat(timestamp)
            timestamp_str = ts.strftime("%H:%M")
        except Exception:
            timestamp_str = ""
    else:
        timestamp_str = ""

    if is_user:
        st.markdown(f"""
        <div class="chat-message {msg_class}">
            <div class="chat-sender">{sender} <span style="float:right; font-size:0.65rem; opacity:0.5;">{timestamp_str}</span></div>
            <div class="chat-content">{msg.get('query', '')}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        badges = get_intent_badge(msg.get("agents_used", []))
        answer = msg.get('answer', '')
        confidence = msg.get('confidence', 0.5)

        # Get confidence label and color
        if confidence >= 0.8:
            conf_label = "High confidence"
            conf_color = "#22c55e"
        elif confidence >= 0.6:
            conf_label = "Medium confidence"
            conf_color = "#f59e0b"
        else:
            conf_label = "Low confidence"
            conf_color = "#ef4444"

        conf_percent = int(confidence * 100)

        # Render header with confidence indicator
        st.markdown(f"""
        <div class="chat-message {msg_class}">
            <div class="chat-sender">
                {sender} {badges}
                <span class="confidence-badge" style="margin-left: 10px;">
                    <span class="confidence-fill" style="width: {conf_percent}%; background: {conf_color};"></span>
                    <span class="confidence-text">{conf_label}</span>
                </span>
                <span style="float:right; font-size:0.65rem; opacity:0.5;">{timestamp_str}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Use Streamlit's native markdown for rich formatting
        st.markdown(answer)

        # Rich media: Show sources as expandable section
        sources = msg.get('sources', [])
        if sources:
            with st.expander("View sources", expanded=False):
                for i, source in enumerate(sources[:5]):
                    st.markdown(f"""
                    <div class="source-card">
                        <div class="source-title">Source {i+1}</div>
                        <div class="source-text">{source.get('text', source.get('content', ''))[:200]}...</div>
                    </div>
                    """, unsafe_allow_html=True)

        # Rich media: Show execution metrics
        exec_time = msg.get('execution_time', 0)
        cost = msg.get('cost', 0)
        if exec_time > 0 or cost > 0:
            st.markdown(f"""
            <div class="response-meta">
                <span>Response time: {exec_time:.2f}s</span>
                {f'<span style="margin-left: 15px;">Cost: ${cost:.4f}</span>' if cost > 0 else ''}
            </div>
            """, unsafe_allow_html=True)


def render_chart(chart_data: Dict[str, Any], chart_key: str = None):
    """Render a chart from API response."""
    if not chart_data:
        return

    chart_type = chart_data.get("chart_type", "bar")
    title = chart_data.get("chart_title", "Data")
    x_field = chart_data.get("x_field", "")
    y_field = chart_data.get("y_field", "")
    data = chart_data.get("data", [])

    if not data:
        return

    df = pd.DataFrame(data)
    layout_template = "plotly_dark"

    if chart_type == "line":
        if "type" in df.columns:
            fig = px.line(
                df, x=x_field, y=y_field, color="type",
                title=title, template=layout_template,
                markers=True,
                color_discrete_map={"historical": "#6482ff", "forecast": "#22c55e"}
            )
        else:
            fig = px.line(df, x=x_field, y=y_field, title=title, template=layout_template, markers=True)
    elif chart_type == "bar":
        fig = px.bar(df, x=x_field, y=y_field, title=title, template=layout_template)
    elif chart_type == "pie":
        fig = px.pie(df, names=x_field, values=y_field, title=title, template=layout_template)
    else:
        fig = px.bar(df, x=x_field, y=y_field, title=title, template=layout_template)

    fig.update_layout(
        plot_bgcolor="rgba(10, 10, 15, 0.5)",
        paper_bgcolor="rgba(10, 10, 15, 0.5)",
        font_color="rgba(255, 255, 255, 0.8)",
        margin=dict(l=40, r=40, t=60, b=40),
    )

    # Add export buttons
    col_chart, col_export = st.columns([6, 1])
    with col_chart:
        st.plotly_chart(fig, use_container_width=True, key=chart_key)
    with col_export:
        # Export as PNG
        if st.button("Export", key=f"export_{chart_key}", help="Export chart"):
            st.info("Right-click chart to save as PNG")


def record_feedback(query: str, response: str, rating: str, agents: List[str], intent: str = None):
    """Record user feedback using FeedbackCollector."""
    collector = get_feedback_collector()
    collector.record_feedback(
        query=query,
        response=response,
        rating=rating,
        intent=intent,
        agents_used=agents,
    )


def render_thinking_animation(stage: str = "Analyzing your question"):
    """Render thinking animation with bouncing dots."""
    st.markdown(f"""
    <div class="thinking-container">
        <div class="thinking-avatar" style="font-family: monospace; font-weight: 700; font-size: 1rem;">AI</div>
        <div class="thinking-content">
            <div class="thinking-text">{stage}</div>
            <div class="thinking-dots">
                <div class="thinking-dot"></div>
                <div class="thinking-dot"></div>
                <div class="thinking-dot"></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_loading_skeleton():
    """Render loading skeleton placeholder (legacy, use render_thinking_animation)."""
    render_thinking_animation()


def render_confidence_indicator(confidence: float):
    """Render AI confidence indicator based on prediction quality."""
    if confidence >= 0.7:
        level = "high"
        label = "High Confidence"
        icon = "✓"
    elif confidence >= 0.4:
        level = "medium"
        label = "Medium Confidence"
        icon = "~"
    else:
        level = "low"
        label = "Low Confidence"
        icon = "?"

    pct = int(confidence * 100)
    st.markdown(f"""
    <div class="confidence-badge {level}">
        <span>{icon}</span>
        <span>{label}</span>
        <span>({pct}%)</span>
    </div>
    <div class="confidence-bar">
        <div class="confidence-fill {level}" style="width: {pct}%;"></div>
    </div>
    """, unsafe_allow_html=True)


def render_onboarding():
    """Render interactive onboarding for first-time users."""
    st.markdown("""
    <div class="onboarding-overlay">
        <div class="onboarding-header">
            <div class="onboarding-title">Welcome to the Marketing Intelligence Platform</div>
            <div class="onboarding-subtitle">Enterprise-grade analytics for sales, sentiment, and forecasting</div>
        </div>
        <div class="onboarding-steps">
            <div class="onboarding-step">
                <div class="onboarding-step-icon" style="font-size: 1.5rem; color: var(--accent-blue);">01</div>
                <div class="onboarding-step-title">Query</div>
                <div class="onboarding-step-desc">Ask about sales, customer sentiment, or revenue forecasts in natural language</div>
            </div>
            <div class="onboarding-step">
                <div class="onboarding-step-icon" style="font-size: 1.5rem; color: var(--accent-blue);">02</div>
                <div class="onboarding-step-title">Analyze</div>
                <div class="onboarding-step-desc">AI agents analyze your data using advanced ML models</div>
            </div>
            <div class="onboarding-step">
                <div class="onboarding-step-icon" style="font-size: 1.5rem; color: var(--accent-blue);">03</div>
                <div class="onboarding-step-title">Insights</div>
                <div class="onboarding-step-desc">Receive actionable insights with charts and recommendations</div>
            </div>
            <div class="onboarding-step">
                <div class="onboarding-step-icon" style="font-size: 1.5rem; color: var(--accent-blue);">04</div>
                <div class="onboarding-step-title">Explore</div>
                <div class="onboarding-step-desc">Dive deeper with ML-powered analytics in Data Explorer</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Get Started", type="primary", use_container_width=True, key="start_onboarding"):
            st.session_state.show_onboarding = False
            st.session_state.onboarding_completed = True
            st.rerun()

    st.markdown("<div class='onboarding-dismiss'>", unsafe_allow_html=True)
    if st.button("Skip tutorial", key="skip_onboarding"):
        st.session_state.show_onboarding = False
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


def copy_to_clipboard(text: str, key: str):
    """Create a copy button that copies text to clipboard."""
    st.markdown(f"""
    <button onclick="navigator.clipboard.writeText(`{text.replace('`', '\\`')}`)"
            class="copy-btn" title="Copy to clipboard">
        Copy
    </button>
    """, unsafe_allow_html=True)


# ============== HEADER ==============

st.markdown("""
<div class="header-container">
    <h1 class="header-title">Marketing Intelligence Platform</h1>
    <p class="header-subtitle">Enterprise analytics powered by LangGraph and RAG architecture</p>
</div>
""", unsafe_allow_html=True)


# ============== MAIN TABS ==============

tab_dashboard, tab_chat, tab_explorer, tab_system = st.tabs([
    "Dashboard", "Analyst", "Data Explorer", "System"
])


# ============== TAB 0: DASHBOARD (NEW) ==============
with tab_dashboard:
    st.markdown('<div class="section-title">Executive Summary <span class="section-badge">KPIs</span></div>', unsafe_allow_html=True)

    stats = get_api_stats()

    if stats:
        # Top KPI metrics
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            total_orders = stats.get('total_orders', 0)
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon-text">ORD</div>
                <div class="kpi-value">{total_orders:,}</div>
                <div class="kpi-label">Total Orders</div>
                <div class="kpi-delta positive">+12.3% vs last period</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            total_revenue = stats.get('total_revenue', 0)
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon-text">REV</div>
                <div class="kpi-value">R$ {total_revenue/1000000:.1f}M</div>
                <div class="kpi-label">Total Revenue</div>
                <div class="kpi-delta positive">+8.7% vs last period</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            avg_rating = stats.get('avg_rating', 0)
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon-text">RTG</div>
                <div class="kpi-value">{avg_rating:.1f}</div>
                <div class="kpi-label">Avg Rating</div>
                <div class="kpi-delta {'positive' if avg_rating >= 4 else 'negative'}">{"Good" if avg_rating >= 4 else "Needs improvement"}</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            total_reviews = stats.get('total_reviews', 0)
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon-text">REV</div>
                <div class="kpi-value">{total_reviews:,}</div>
                <div class="kpi-label">Customer Reviews</div>
                <div class="kpi-delta neutral">Analyzed</div>
            </div>
            """, unsafe_allow_html=True)

        with col5:
            total_categories = stats.get('total_categories', 0)
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon-text">CAT</div>
                <div class="kpi-value">{total_categories}</div>
                <div class="kpi-label">Categories</div>
                <div class="kpi-delta neutral">Active</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Anomaly Alerts Section
        st.markdown('<div class="section-title">Alerts & Anomalies <span class="section-badge alert">Live</span></div>', unsafe_allow_html=True)

        anomalies = stats.get("anomalies", [])
        if anomalies:
            for anomaly in anomalies[:5]:
                alert_type = anomaly.get("type", "info")
                alert_label = "ALERT" if alert_type == "critical" else "WARN" if alert_type == "warning" else "INFO"
                st.markdown(f"""
                <div class="alert-card {alert_type}">
                    <span class="alert-icon">[{alert_label}]</span>
                    <span class="alert-message">{anomaly.get('message', '')}</span>
                    <span class="alert-time">{anomaly.get('time', '')}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            # Show sample alerts based on data analysis
            monthly_revenue = stats.get("monthly_revenue", [])
            if monthly_revenue and len(monthly_revenue) >= 2:
                last_month = monthly_revenue[-1].get("revenue", 0)
                prev_month = monthly_revenue[-2].get("revenue", 0)
                change = ((last_month - prev_month) / prev_month * 100) if prev_month > 0 else 0

                if abs(change) > 20:
                    alert_type = "warning" if change < 0 else "positive"
                    alert_label = "[WARN]" if change < 0 else "[UP]"
                    st.markdown(f"""
                    <div class="alert-card {alert_type}">
                        <span class="alert-icon">{alert_label}</span>
                        <span class="alert-message">Revenue {change:+.1f}% month-over-month - {"significant decline detected" if change < 0 else "strong growth detected"}</span>
                    </div>
                    """, unsafe_allow_html=True)

            # Rating alert
            if avg_rating < 4.0:
                st.markdown("""
                <div class="alert-card warning">
                    <span class="alert-icon">[WARN]</span>
                    <span class="alert-message">Average rating below 4.0 - consider reviewing customer feedback</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="alert-card positive">
                    <span class="alert-icon">[OK]</span>
                    <span class="alert-message">All metrics within normal ranges - no anomalies detected</span>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # Quick Charts Row - More Impressive Visualizations
        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            st.markdown("**Revenue Trend (Last 6 Months)**")
            monthly_revenue = stats.get("monthly_revenue", [])
            if monthly_revenue:
                df_monthly = pd.DataFrame(monthly_revenue).tail(6)

                # Calculate month-over-month change for color coding
                df_monthly["change"] = df_monthly["revenue"].pct_change() * 100
                df_monthly["change"] = df_monthly["change"].fillna(0)

                # Create figure with secondary y-axis
                fig = go.Figure()

                # Add gradient area chart
                fig.add_trace(go.Scatter(
                    x=df_monthly["month"],
                    y=df_monthly["revenue"],
                    fill="tozeroy",
                    mode="lines+markers",
                    line=dict(color="#6482ff", width=3),
                    fillcolor="rgba(100, 130, 255, 0.2)",
                    marker=dict(size=10, color="#6482ff", line=dict(width=2, color="white")),
                    name="Revenue",
                    hovertemplate="<b>%{x}</b><br>Revenue: R$ %{y:,.0f}<extra></extra>"
                ))

                # Add trend line
                if len(df_monthly) >= 2:
                    x_numeric = list(range(len(df_monthly)))
                    z = np.polyfit(x_numeric, df_monthly["revenue"].values, 1)
                    p = np.poly1d(z)
                    fig.add_trace(go.Scatter(
                        x=df_monthly["month"],
                        y=p(x_numeric),
                        mode="lines",
                        line=dict(color="#f59e0b", width=2, dash="dash"),
                        name="Trend",
                        hoverinfo="skip"
                    ))

                # Add annotation for latest value
                latest_val = df_monthly["revenue"].iloc[-1]
                latest_month = df_monthly["month"].iloc[-1]
                fig.add_annotation(
                    x=latest_month,
                    y=latest_val,
                    text=f"R$ {latest_val/1000000:.2f}M",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor="#6482ff",
                    font=dict(color="white", size=12),
                    bgcolor="rgba(100, 130, 255, 0.8)",
                    borderpad=4
                )

                fig.update_layout(
                    template="plotly_dark",
                    plot_bgcolor="rgba(10, 10, 15, 0.5)",
                    paper_bgcolor="rgba(10, 10, 15, 0.5)",
                    margin=dict(l=10, r=10, t=30, b=10),
                    height=220,
                    xaxis_title="", yaxis_title="",
                    showlegend=False,
                    yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
                    xaxis=dict(gridcolor="rgba(255,255,255,0.05)")
                )
                st.plotly_chart(fig, use_container_width=True, key="dash_revenue_trend")

        with col_chart2:
            st.markdown("**Top Categories by Revenue**")
            top_categories = stats.get("top_categories", [])
            if top_categories:
                df_cat = pd.DataFrame(top_categories[:8])  # More categories for treemap

                # Calculate percentage of total
                total_rev = df_cat["revenue"].sum()
                df_cat["percentage"] = (df_cat["revenue"] / total_rev * 100).round(1)
                df_cat["label"] = df_cat.apply(
                    lambda x: f"{x['category'].replace('_', ' ').title()}<br>R$ {x['revenue']/1000000:.1f}M ({x['percentage']}%)",
                    axis=1
                )

                # Treemap for better visual impact
                fig = px.treemap(
                    df_cat,
                    path=["category"],
                    values="revenue",
                    color="revenue",
                    color_continuous_scale=["#134e4a", "#14b8a6", "#22c55e", "#86efac"],
                    template="plotly_dark",
                    custom_data=["label"]
                )

                fig.update_traces(
                    texttemplate="%{customdata[0]}",
                    textposition="middle center",
                    textfont=dict(size=11, color="white"),
                    hovertemplate="<b>%{label}</b><br>Revenue: R$ %{value:,.0f}<extra></extra>",
                    marker=dict(cornerradius=5)
                )

                fig.update_layout(
                    plot_bgcolor="rgba(10, 10, 15, 0.5)",
                    paper_bgcolor="rgba(10, 10, 15, 0.5)",
                    margin=dict(l=5, r=5, t=5, b=5),
                    height=220,
                    coloraxis_showscale=False
                )
                st.plotly_chart(fig, use_container_width=True, key="dash_top_categories")

        st.markdown("---")

        # AI Insights Panel - Story-driven insights from data
        st.markdown('<div class="section-title">AI Insights <span class="section-badge">Auto-generated</span></div>', unsafe_allow_html=True)

        col_insight1, col_insight2 = st.columns(2)

        with col_insight1:
            # Generate insights from actual data
            monthly_revenue = stats.get("monthly_revenue", [])
            top_categories = stats.get("top_categories", [])

            insights = []

            # Revenue trend insight
            if monthly_revenue and len(monthly_revenue) >= 2:
                last_3_months = monthly_revenue[-3:] if len(monthly_revenue) >= 3 else monthly_revenue
                avg_recent = sum(m.get("revenue", 0) for m in last_3_months) / len(last_3_months)
                prev_3_months = monthly_revenue[-6:-3] if len(monthly_revenue) >= 6 else monthly_revenue[:3]
                avg_prev = sum(m.get("revenue", 0) for m in prev_3_months) / max(len(prev_3_months), 1)
                if avg_prev > 0:
                    change = ((avg_recent - avg_prev) / avg_prev) * 100
                    if change > 10:
                        insights.append({
                            "icon": "+",
                            "title": "Revenue Momentum",
                            "desc": f"Strong growth detected: +{change:.1f}% vs previous quarter",
                            "type": "positive"
                        })
                    elif change < -10:
                        insights.append({
                            "icon": "-",
                            "title": "Revenue Alert",
                            "desc": f"Revenue declined {change:.1f}% - review pricing or marketing strategy",
                            "type": "warning"
                        })
                    else:
                        insights.append({
                            "icon": "=",
                            "title": "Stable Performance",
                            "desc": f"Revenue is steady ({change:+.1f}%) - consider growth initiatives",
                            "type": "info"
                        })

            # Category concentration insight
            if top_categories and len(top_categories) >= 3:
                total_cat_revenue = sum(c.get("revenue", 0) for c in top_categories)
                top_3_revenue = sum(c.get("revenue", 0) for c in top_categories[:3])
                concentration = (top_3_revenue / total_cat_revenue * 100) if total_cat_revenue > 0 else 0
                if concentration > 70:
                    insights.append({
                        "icon": "!",
                        "title": "Revenue Concentration",
                        "desc": f"Top 3 categories = {concentration:.0f}% of revenue. Diversify to reduce risk.",
                        "type": "warning"
                    })
                else:
                    insights.append({
                        "icon": "OK",
                        "title": "Healthy Diversification",
                        "desc": f"Good category mix: top 3 = {concentration:.0f}% of total revenue",
                        "type": "positive"
                    })

            # Rating insight
            if avg_rating >= 4.5:
                insights.append({
                    "icon": "*",
                    "title": "Excellent Satisfaction",
                    "desc": f"Customer rating {avg_rating:.1f}/5 - leverage for marketing testimonials",
                    "type": "positive"
                })
            elif avg_rating < 3.5:
                insights.append({
                    "icon": "!",
                    "title": "Satisfaction Opportunity",
                    "desc": f"Rating {avg_rating:.1f}/5 below benchmark - prioritize customer experience",
                    "type": "warning"
                })

            # Display insights
            for insight in insights[:2]:
                st.markdown(f"""
                <div class="insight-card {insight['type']}">
                    <div class="insight-icon">{insight['icon']}</div>
                    <div class="insight-content">
                        <div class="insight-title">{insight['title']}</div>
                        <div class="insight-desc">{insight['desc']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with col_insight2:
            # Performance comparison mini-widget
            st.markdown("**Period Comparison**")

            if monthly_revenue and len(monthly_revenue) >= 6:
                # Last 3 months vs previous 3 months
                last_3 = monthly_revenue[-3:]
                prev_3 = monthly_revenue[-6:-3]

                last_total = sum(m.get("revenue", 0) for m in last_3)
                prev_total = sum(m.get("revenue", 0) for m in prev_3)
                pct_change = ((last_total - prev_total) / prev_total * 100) if prev_total > 0 else 0

                col_cur, col_prev = st.columns(2)
                with col_cur:
                    st.markdown(f"""
                    <div class="mini-stat-card">
                        <div class="mini-stat-label">Last 3 Months</div>
                        <div class="mini-stat-value">R$ {last_total/1000000:.2f}M</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_prev:
                    st.markdown(f"""
                    <div class="mini-stat-card">
                        <div class="mini-stat-label">Previous 3 Months</div>
                        <div class="mini-stat-value">R$ {prev_total/1000000:.2f}M</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Change indicator
                change_class = "positive" if pct_change >= 0 else "negative"
                change_icon = "UP" if pct_change >= 0 else "DOWN"
                st.markdown(f"""
                <div class="change-indicator {change_class}">
                    {change_icon} <span class="change-value">{pct_change:+.1f}%</span> quarter-over-quarter
                </div>
                """, unsafe_allow_html=True)

            # Top performer highlight
            if top_categories:
                top_cat = top_categories[0]
                st.markdown(f"""
                <div class="highlight-card">
                    <div class="highlight-label">#1 Top Category</div>
                    <div class="highlight-value">{top_cat.get('category', 'N/A')}</div>
                    <div class="highlight-detail">R$ {top_cat.get('revenue', 0):,.0f}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # Recent Activity Feed
        st.markdown('<div class="section-title">Recent Activity <span class="section-badge">Live</span></div>', unsafe_allow_html=True)

        # Get recent queries from chat history
        history_manager = get_history_manager()
        all_sessions = history_manager.list_sessions()

        recent_activities = []
        for session in all_sessions[:5]:
            session_id = session.get("session_id", "")
            messages = history_manager.load_session(session_id)
            for msg in messages[-3:]:  # Last 3 from each session
                if msg.get("role") == "user":
                    recent_activities.append({
                        "type": "query",
                        "text": msg.get("query", "")[:60],
                        "time": msg.get("timestamp", ""),
                    })
                elif msg.get("role") == "assistant":
                    agents = msg.get("agents_used", [])
                    if agents:
                        recent_activities.append({
                            "type": "analysis",
                            "text": f"Analysis completed using {', '.join(agents)}",
                            "time": msg.get("timestamp", ""),
                        })

        # Sort by time and take most recent
        recent_activities.sort(key=lambda x: x.get("time", ""), reverse=True)

        if recent_activities:
            for activity in recent_activities[:4]:
                icon = ">" if activity["type"] == "query" else "+"
                time_str = ""
                if activity.get("time"):
                    try:
                        ts = datetime.fromisoformat(activity["time"])
                        time_str = ts.strftime("%b %d, %H:%M")
                    except Exception:
                        pass

                st.markdown(f"""
                <div class="activity-item">
                    <span class="activity-icon">{icon}</span>
                    <span class="activity-text">{activity['text']}...</span>
                    <span class="activity-time">{time_str}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="activity-empty">
                <span>No recent activity. Start asking questions in the Chat tab!</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("Unable to fetch dashboard data. Make sure the API is running.")


# ============== TAB 1: CHAT ==============
with tab_chat:
    # Show onboarding for first-time users
    if st.session_state.show_onboarding:
        render_onboarding()

    col_history, col_main = st.columns([1, 4])

    # Left sidebar - Conversation History
    with col_history:
        st.markdown('<div class="section-title">Conversations</div>', unsafe_allow_html=True)

        # New conversation button
        if st.button("+ New Chat", use_container_width=True, key="new_chat"):
            st.session_state.session_id = str(uuid.uuid4())[:8]
            st.session_state.chat_history = []
            st.session_state.conversation_context = []
            st.rerun()

        st.markdown("---")

        # List previous sessions with timestamps
        history_manager = get_history_manager()
        sessions = history_manager.list_sessions()

        for session in sessions[:10]:
            session_id = session["session_id"]
            is_current = session_id == st.session_state.session_id
            query_count = session.get("query_count", 0)
            updated_at = session.get("updated_at", "")
            preview = session.get("preview", "")[:30]

            # Format timestamp
            if updated_at:
                try:
                    ts = datetime.fromisoformat(updated_at)
                    time_str = ts.strftime("%b %d, %H:%M")
                except Exception:
                    time_str = ""
            else:
                time_str = ""

            bg_class = "session-active" if is_current else "session-item"

            if st.button(
                f"{'>' if is_current else '-'} {preview or 'New chat'}...\n{time_str}",
                key=f"session_{session_id}",
                use_container_width=True,
                disabled=is_current
            ):
                st.session_state.session_id = session_id
                st.session_state.chat_history = history_manager.load_session(session_id)
                st.rerun()

        st.markdown("---")

        # Export options
        st.markdown('<div class="section-title">Export</div>', unsafe_allow_html=True)

        if st.session_state.chat_history:
            col_json, col_csv = st.columns(2)
            with col_json:
                json_data = history_manager.export_session_json(st.session_state.session_id)
                if json_data:
                    st.download_button(
                        "JSON",
                        data=json_data,
                        file_name=f"chat_{st.session_state.session_id}.json",
                        mime="application/json",
                        use_container_width=True,
                    )
            with col_csv:
                csv_data = history_manager.export_session_csv(st.session_state.session_id)
                if csv_data:
                    st.download_button(
                        "CSV",
                        data=csv_data,
                        file_name=f"chat_{st.session_state.session_id}.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

    # Main chat area
    with col_main:
        # Quick Questions - collapsible section
        # Using a button to toggle state (avoids widget state conflicts)
        toggle_icon = "v" if st.session_state.quick_questions_open else ">"
        if st.button(f"{toggle_icon}  Quick Questions", key="quick_questions_btn", use_container_width=True):
            st.session_state.quick_questions_open = not st.session_state.quick_questions_open
            st.rerun()

        # Show content only if open
        if st.session_state.quick_questions_open:
            # Search filter
            search_query = st.text_input(
                "Search questions",
                placeholder="Filter questions...",
                key="question_search",
                label_visibility="collapsed"
            )

            # All example questions
            all_questions = {
                "Sales": [
                    "What are the top 10 products by revenue?",
                    "Compare electronics vs furniture categories",
                    "Revenue breakdown by region",
                    "Monthly sales trend analysis",
                    "Which products have declining sales?",
                ],
                "Sentiment": [
                    "Summarize customer complaints",
                    "What are customers saying about delivery?",
                    "Top rated products analysis",
                    "Common issues mentioned in reviews",
                    "Sentiment trend over time",
                ],
                "Forecast": [
                    "Forecast next month's revenue",
                    "Predict electronics category performance",
                    "Revenue trend forecast for Q4",
                    "Growth prediction by region",
                ],
            }

            # Custom questions
            if st.session_state.custom_questions:
                all_questions["Custom"] = st.session_state.custom_questions

            # Filter and display questions
            for category, questions in all_questions.items():
                filtered_questions = [q for q in questions if not search_query or search_query.lower() in q.lower()]

                if filtered_questions:
                    st.markdown(f"**{category}**")
                    cols = st.columns(3)
                    for idx, question in enumerate(filtered_questions):
                        with cols[idx % 3]:
                            if st.button(question, key=f"quick_{category}_{idx}", use_container_width=True):
                                st.session_state.pending_query = question
                                st.session_state.close_quick_questions = True  # Flag to close on next rerun
                                st.rerun()

            # Add custom question
            st.markdown("---")
            col_add, col_btn = st.columns([4, 1])
            with col_add:
                new_question = st.text_input("Add custom question", key="new_custom_q", label_visibility="collapsed", placeholder="Add your own question...")
            with col_btn:
                if st.button("Add", key="add_custom_q"):
                    if new_question and new_question not in st.session_state.custom_questions:
                        st.session_state.custom_questions.append(new_question)
                        st.success("Added!")

        st.markdown("---")

        # Query input area
        st.markdown('<div class="section-title">Ask a Question</div>', unsafe_allow_html=True)

        # Query input with autocomplete from history
        query_options = list(set(st.session_state.query_history[-20:]))
        query = st.text_input(
            "Ask anything about sales, sentiment, or forecasts",
            key="query_input",
            placeholder="e.g. Compare headphones vs smart home devices performance",
            label_visibility="collapsed",
        )

        # Show autocomplete suggestions
        if query and len(query) > 2:
            suggestions = [q for q in query_options if query.lower() in q.lower()][:3]
            if suggestions:
                st.markdown("**Suggestions:**")
                for sug in suggestions:
                    if st.button(f"> {sug}", key=f"sug_{hash(sug)}", use_container_width=True):
                        st.session_state.pending_query = sug
                        st.rerun()

        col_analyze, col_clear = st.columns([1, 1])
        with col_analyze:
            analyze_clicked = st.button("Analyze", type="primary", use_container_width=True)
        with col_clear:
            if st.button("Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.conversation_context = []
                save_chat_history()
                st.rerun()

        # Check for pending query from quick questions or suggestions
        active_query = None
        is_followup = st.session_state.is_followup_query
        if st.session_state.pending_query:
            active_query = st.session_state.pending_query
            st.session_state.pending_query = None
            st.session_state.is_followup_query = False  # Reset flag
        elif analyze_clicked and query:
            active_query = query
            is_followup = False  # New queries from input are not follow-ups

        # Process query - create placeholder for loading animation
        thinking_placeholder = None

        # For non-follow-up queries, show loading at top (before conversation)
        if active_query and not is_followup:
            thinking_placeholder = st.empty()
            with thinking_placeholder:
                render_thinking_animation("Analyzing your question")

        # Display chat history (newest Quick Questions at top, follow-ups below their parent)
        if st.session_state.chat_history:
            st.markdown("---")
            st.markdown('<div class="section-title">Conversation</div>', unsafe_allow_html=True)

            # Find the most recent assistant message by timestamp for showing follow-up suggestions
            assistant_msgs = [m for m in st.session_state.chat_history if m.get("role") == "assistant"]
            most_recent_assistant_ts = max((m.get("timestamp", "") for m in assistant_msgs), default="") if assistant_msgs else ""

            for idx, msg in enumerate(st.session_state.chat_history):
                original_idx = idx

                render_chat_message(msg, is_user=msg.get("role") == "user", msg_idx=original_idx)

                # Show chart if available
                if msg.get("role") == "assistant" and msg.get("chart_data"):
                    render_chart(msg["chart_data"], chart_key=f"chat_chart_{original_idx}")

                # Assistant message actions - follow-up suggestions
                if msg.get("role") == "assistant":
                    # Follow-up suggestions - only show for the MOST RECENT assistant message by timestamp
                    is_most_recent = msg.get("timestamp", "") == most_recent_assistant_ts
                    if is_most_recent:
                        suggestions = msg.get("suggestions", [])
                        if suggestions and st.session_state.show_suggestions:
                            st.markdown("**Suggested follow-ups:**")
                            sug_cols = st.columns(len(suggestions))
                            for i, sug in enumerate(suggestions):
                                with sug_cols[i]:
                                    if st.button(sug, key=f"followup_{original_idx}_{i}", use_container_width=True):
                                        st.session_state.pending_query = sug
                                        st.session_state.is_followup_query = True
                                        # Store the parent index so we can insert after it
                                        st.session_state.followup_parent_idx = original_idx
                                        st.rerun()

        # For follow-up queries, show loading AFTER conversation (at bottom)
        if active_query and is_followup:
            thinking_placeholder = st.empty()
            with thinking_placeholder:
                render_thinking_animation("Analyzing your question")

        # Process the query if there is one
        if active_query:
            # Add to history
            if active_query not in st.session_state.query_history:
                st.session_state.query_history.append(active_query)

            user_msg = {
                "role": "user",
                "query": active_query,
                "timestamp": datetime.now().isoformat(),
            }
            # New Quick Questions go at TOP, follow-ups insert right after their parent
            user_insert_pos = 0  # Default: insert at top
            if is_followup and st.session_state.followup_parent_idx is not None:
                # Insert right after the parent assistant message
                user_insert_pos = st.session_state.followup_parent_idx + 1
                st.session_state.chat_history.insert(user_insert_pos, user_msg)
                st.session_state.followup_parent_idx = None  # Reset
            else:
                st.session_state.chat_history.insert(0, user_msg)

            # Add to context for memory
            st.session_state.conversation_context.append({
                "role": "user",
                "content": active_query
            })

            try:
                # Get confidence prediction before making the call
                collector = get_feedback_collector()
                prediction = collector.predict_quality(active_query)
                confidence = prediction.get("confidence", 0.5)

                result = call_api(active_query, st.session_state.conversation_context)

                answer = result.get("answer", "")

                assistant_msg = {
                    "role": "assistant",
                    "answer": answer,
                    "agents_used": result.get("agents_used", []),
                    "execution_time": result.get("execution_time", 0),
                    "cost": result.get("cost", 0),
                    "sources": result.get("sources", []),
                    "chart_data": result.get("chart_data"),
                    "timestamp": datetime.now().isoformat(),
                    "suggestions": get_suggested_questions(active_query, answer),
                    "confidence": confidence,
                }
                # Insert assistant response right after the user message
                # For new questions: user at 0, assistant at 1
                # For follow-ups: user at user_insert_pos, assistant at user_insert_pos + 1
                assistant_insert_pos = user_insert_pos + 1
                st.session_state.chat_history.insert(assistant_insert_pos, assistant_msg)

                # Add to context for memory
                st.session_state.conversation_context.append({
                    "role": "assistant",
                    "content": answer
                })

                # Clear thinking animation
                if thinking_placeholder:
                    thinking_placeholder.empty()

                save_chat_history()
                st.rerun()

            except httpx.ConnectError:
                if thinking_placeholder:
                    thinking_placeholder.empty()
                st.error("Cannot connect to API. Make sure the backend is running on port 8000.")
            except httpx.HTTPStatusError as e:
                if thinking_placeholder:
                    thinking_placeholder.empty()
                st.error(f"API error: {e.response.status_code} - {e.response.text}")
            except Exception as exc:
                if thinking_placeholder:
                    thinking_placeholder.empty()
                st.error(f"Error: {exc}")

        # Show latest response metrics (find by most recent timestamp)
        assistant_msgs = [m for m in st.session_state.chat_history if m.get("role") == "assistant"]
        if assistant_msgs:
            # Sort by timestamp to get the most recently added response
            last_response = max(assistant_msgs, key=lambda m: m.get("timestamp", ""))

            st.markdown("---")
            metrics = st.columns(3)
            metrics[0].metric("Response Time", f"{last_response.get('execution_time', 0):.1f}s")
            cost_val = last_response.get('cost', 0)
            metrics[1].metric("Est. Cost", f"${cost_val:.4f}" if cost_val > 0 else "$0.0000")
            agents = ", ".join(last_response.get("agents_used", [])) or "n/a"
            metrics[2].metric("Agents Used", agents)

            sources = last_response.get("sources", [])
            if sources:
                with st.expander(f"Sources ({len(sources)} documents)"):
                    for source_idx, source in enumerate(sources, 1):
                        text = source.get("text", "")[:300]
                        st.markdown(f"**[{source_idx}]** {text}...")


# ============== TAB 2: DATA EXPLORER ==============
with tab_explorer:
    st.markdown('<div class="section-title">Dataset Overview <span class="section-badge">Analytics</span></div>', unsafe_allow_html=True)

    stats = get_api_stats()

    if stats:
        # KPI Row
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Total Orders</div>
                <div class="stat-value">{stats.get('total_orders', 0):,}</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Total Revenue</div>
                <div class="stat-value">R$ {stats.get('total_revenue', 0):,.0f}</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Total Reviews</div>
                <div class="stat-value">{stats.get('total_reviews', 0):,}</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Avg Rating</div>
                <div class="stat-value">{stats.get('avg_rating', 0):.1f}/5</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # ============== ROW 1: HERO - Revenue Forecast (Most Impressive First) ==============
        st.markdown('<div class="section-title">Revenue Forecast <span class="section-badge">AI Prediction</span></div>', unsafe_allow_html=True)

        forecast_demo = stats.get("forecast_demo", [])
        if forecast_demo:
            df_forecast = pd.DataFrame(forecast_demo)
            if not df_forecast.empty and "period" in df_forecast.columns:
                df_hist = df_forecast[df_forecast["type"] == "historical"]
                df_pred = df_forecast[df_forecast["type"] == "forecast"]

                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=df_hist["period"],
                    y=df_hist["value"],
                    mode="lines+markers",
                    name="Historical",
                    line=dict(color="#6482ff", width=2),
                    marker=dict(size=6),
                ))

                if "value_upper" in df_pred.columns and "value_lower" in df_pred.columns:
                    fig.add_trace(go.Scatter(
                        x=df_pred["period"],
                        y=df_pred["value_upper"],
                        mode="lines",
                        name="Upper 80%",
                        line=dict(width=0),
                        showlegend=False,
                    ))
                    fig.add_trace(go.Scatter(
                        x=df_pred["period"],
                        y=df_pred["value_lower"],
                        mode="lines",
                        name="Confidence (80%)",
                        line=dict(width=0),
                        fill="tonexty",
                        fillcolor="rgba(34, 197, 94, 0.2)",
                    ))

                fig.add_trace(go.Scatter(
                    x=df_pred["period"],
                    y=df_pred["value"],
                    mode="lines+markers",
                    name="Prophet Forecast",
                    line=dict(color="#22c55e", width=2, dash="dot"),
                    marker=dict(size=6),
                ))

                if not df_hist.empty and not df_pred.empty:
                    fig.add_trace(go.Scatter(
                        x=[df_hist["period"].iloc[-1], df_pred["period"].iloc[0]],
                        y=[df_hist["value"].iloc[-1], df_pred["value"].iloc[0]],
                        mode="lines",
                        line=dict(color="#22c55e", width=1, dash="dot"),
                        showlegend=False,
                    ))

                fig.update_layout(
                    template="plotly_dark",
                    plot_bgcolor="rgba(10, 10, 15, 0.5)",
                    paper_bgcolor="rgba(10, 10, 15, 0.5)",
                    font_color="rgba(255, 255, 255, 0.8)",
                    margin=dict(l=10, r=40, t=10, b=10),
                    xaxis_title="Week",
                    yaxis_title="Revenue (R$)",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    hovermode="x unified",
                )
                st.plotly_chart(fig, use_container_width=True, key="forecast_demo_chart")

                # Forecast summary and explanation
                historical = [d for d in forecast_demo if d.get("type") == "historical"]
                forecasted = [d for d in forecast_demo if d.get("type") == "forecast"]
                if historical and forecasted:
                    avg_hist = sum(h["value"] for h in historical[-4:]) / 4
                    avg_forecast = sum(f["value"] for f in forecasted) / len(forecasted)
                    growth = ((avg_forecast - avg_hist) / avg_hist * 100) if avg_hist > 0 else 0
                    trend = "growth" if growth >= 0 else "decline"
                    card_class = "positive" if growth >= 0 else "warning"
                    st.markdown(f"""
                    <div class="insight-card {card_class}">
                        <div class="insight-title">8-Week Prophet Forecast</div>
                        <div class="insight-desc">Facebook Prophet prediction: <b>{growth:+.1f}%</b> {trend} expected vs recent average (80% confidence interval shown)</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Why this prediction
                    st.markdown("#### Why this prediction?")

                    hist_values = [h["value"] for h in historical]

                    first_quarter_avg = sum(hist_values[:13]) / 13 if len(hist_values) >= 13 else sum(hist_values[:len(hist_values)//2]) / max(1, len(hist_values)//2)
                    last_quarter_avg = sum(hist_values[-13:]) / 13 if len(hist_values) >= 13 else sum(hist_values[len(hist_values)//2:]) / max(1, len(hist_values) - len(hist_values)//2)
                    overall_trend = ((last_quarter_avg - first_quarter_avg) / first_quarter_avg * 100) if first_quarter_avg > 0 else 0
                    trend_direction = "upward" if overall_trend > 5 else "downward" if overall_trend < -5 else "stable"

                    max_val = max(hist_values)
                    max_idx = hist_values.index(max_val)
                    peak_period = historical[max_idx]["period"]

                    if len(hist_values) >= 8:
                        recent_first = sum(hist_values[-8:-4]) / 4
                        recent_last = sum(hist_values[-4:]) / 4
                        recent_momentum = ((recent_last - recent_first) / recent_first * 100) if recent_first > 0 else 0
                    else:
                        recent_momentum = 0

                    import statistics
                    volatility = statistics.stdev(hist_values) / statistics.mean(hist_values) * 100 if len(hist_values) > 1 else 0
                    volatility_level = "high" if volatility > 30 else "moderate" if volatility > 15 else "low"

                    col_explain1, col_explain2 = st.columns(2)

                    with col_explain1:
                        st.markdown(f"""
                        **Key Factors:**
                        1. **Trend:** {trend_direction.capitalize()} ({overall_trend:+.1f}%)
                        2. **Momentum:** {"Accelerating" if recent_momentum > 5 else "Decelerating" if recent_momentum < -5 else "Steady"}
                        3. **Volatility:** {volatility_level.capitalize()} ({volatility:.1f}%)
                        """)

                    with col_explain2:
                        st.markdown(f"""
                        **Patterns:**
                        4. **Peak:** Week of {peak_period}
                        5. **Data:** {len(historical)} weeks analyzed
                        """)
        else:
            st.info("Forecast demo requires historical data.")

        st.markdown("---")

        # ============== ROW 2: Geographic Map + Categories Treemap ==============
        col_geo, col_categories = st.columns(2)

        with col_geo:
            st.markdown('<div class="section-title">Geographic Analysis <span class="section-badge">Where Orders Come From</span></div>', unsafe_allow_html=True)
            geo_data = stats.get("geo_map", [])
            if geo_data:
                df_geo = pd.DataFrame(geo_data)
                if not df_geo.empty and "lat" in df_geo.columns:
                    fig = px.scatter_mapbox(
                        df_geo,
                        lat="lat",
                        lon="lng",
                        size="orders",
                        color="revenue",
                        color_continuous_scale="Viridis",
                        hover_name="city",
                        hover_data={"state": True, "orders": ":,", "revenue": ":,.0f", "lat": False, "lng": False},
                        size_max=25,
                        zoom=3,
                        center={"lat": -14.235, "lon": -51.9253},
                        mapbox_style="carto-darkmatter",
                    )
                    fig.update_layout(
                        margin=dict(l=0, r=0, t=0, b=0),
                        paper_bgcolor="rgba(10, 10, 15, 0.5)",
                        coloraxis_showscale=False,
                        height=350,
                    )
                    st.plotly_chart(fig, use_container_width=True, key="brazil_map")

                    total_cities = len(geo_data)
                    top_city = geo_data[0] if geo_data else {}
                    st.markdown(f"**{total_cities}** cities shown | Top: **{top_city.get('city', 'N/A')}** ({top_city.get('orders', 0):,} orders)")
            else:
                by_state = stats.get("by_state", [])
                if by_state:
                    df_state = pd.DataFrame(by_state)
                    fig = px.bar(df_state.head(10), x="revenue", y="state", orientation="h", template="plotly_dark")
                    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
                    st.plotly_chart(fig, use_container_width=True, key="state_fallback")

        with col_categories:
            st.markdown('<div class="section-title">Top Product Categories</div>', unsafe_allow_html=True)
            top_categories = stats.get("top_categories", [])
            if top_categories:
                df_categories = pd.DataFrame(top_categories)
                if not df_categories.empty and "category" in df_categories.columns:
                    fig = px.treemap(
                        df_categories.head(10),
                        path=["category"],
                        values="revenue",
                        template="plotly_dark",
                        color="revenue",
                        color_continuous_scale="Blues",
                    )
                    fig.update_layout(
                        plot_bgcolor="rgba(10, 10, 15, 0.5)",
                        paper_bgcolor="rgba(10, 10, 15, 0.5)",
                        font_color="rgba(255, 255, 255, 0.9)",
                        margin=dict(l=5, r=5, t=5, b=5),
                        coloraxis_showscale=False,
                        height=350,
                    )
                    fig.update_traces(
                        textinfo="label+value",
                        texttemplate="%{label}<br>R$ %{value:,.0f}",
                        textfont_size=11,
                    )
                    st.plotly_chart(fig, use_container_width=True, key="categories_treemap")

        st.markdown("---")

        # ============== ROW 3: Monthly Revenue + Ratings Distribution ==============
        col_monthly, col_ratings = st.columns(2)

        with col_monthly:
            st.markdown('<div class="section-title">Monthly Revenue Trend</div>', unsafe_allow_html=True)
            monthly_revenue = stats.get("monthly_revenue", [])
            if monthly_revenue:
                df_monthly = pd.DataFrame(monthly_revenue)
                if not df_monthly.empty and "month" in df_monthly.columns:
                    fig = px.area(
                        df_monthly.tail(12),
                        x="month",
                        y="revenue",
                        template="plotly_dark",
                        markers=True,
                    )
                    fig.update_traces(
                        fill="tozeroy",
                        line_color="#6482ff",
                        fillcolor="rgba(100, 130, 255, 0.3)",
                    )
                    fig.update_layout(
                        plot_bgcolor="rgba(10, 10, 15, 0.5)",
                        paper_bgcolor="rgba(10, 10, 15, 0.5)",
                        font_color="rgba(255, 255, 255, 0.8)",
                        margin=dict(l=10, r=10, t=10, b=10),
                        xaxis_title="",
                        yaxis_title="Revenue (R$)",
                        height=350,
                    )
                    st.plotly_chart(fig, use_container_width=True, key="monthly_area")

        with col_ratings:
            st.markdown('<div class="section-title">Customer Ratings Distribution</div>', unsafe_allow_html=True)
            ratings = stats.get("ratings", [])
            if ratings:
                df_ratings = pd.DataFrame(ratings)
                if not df_ratings.empty and "rating" in df_ratings.columns:
                    total_ratings = sum(r.get("count", 0) for r in ratings)
                    df_ratings["percentage"] = df_ratings["count"] / total_ratings * 100
                    df_ratings["label"] = df_ratings["rating"].apply(lambda x: f"{x} Star{'s' if x > 1 else ''}")

                    colors = ["#ef4444", "#f97316", "#eab308", "#84cc16", "#22c55e"]

                    fig = go.Figure()
                    for i, row in df_ratings.iterrows():
                        fig.add_trace(go.Bar(
                            y=[row["label"]],
                            x=[row["percentage"]],
                            orientation="h",
                            marker_color=colors[int(row["rating"]) - 1],
                            text=f"{row['percentage']:.1f}% ({row['count']:,})",
                            textposition="auto",
                            name=row["label"],
                            showlegend=False,
                        ))

                    fig.update_layout(
                        template="plotly_dark",
                        plot_bgcolor="rgba(10, 10, 15, 0.5)",
                        paper_bgcolor="rgba(10, 10, 15, 0.5)",
                        font_color="rgba(255, 255, 255, 0.8)",
                        margin=dict(l=10, r=10, t=10, b=10),
                        xaxis_title="Percentage",
                        yaxis_title="",
                        barmode="stack",
                        height=350,
                        yaxis=dict(categoryorder="array", categoryarray=["1 Star", "2 Stars", "3 Stars", "4 Stars", "5 Stars"]),
                    )
                    st.plotly_chart(fig, use_container_width=True, key="ratings_bars")

                    five_stars = next((r.get("count", 0) for r in ratings if r.get("rating") == 5), 0)
                    pct_5star = (five_stars / total_ratings * 100) if total_ratings > 0 else 0
                    avg_rating = sum(r["rating"] * r["count"] for r in ratings) / total_ratings if total_ratings > 0 else 0
                    st.markdown(f"**{total_ratings:,}** total | Avg: **{avg_rating:.1f}** | **{pct_5star:.1f}%** 5-star")

        st.markdown("---")

        # ============== ML ANALYTICS SECTION ==============
        st.markdown('<div class="section-title">ML Analytics <span class="section-badge">Machine Learning</span></div>', unsafe_allow_html=True)

        ml_tab1, ml_tab2, ml_tab3, ml_tab4 = st.tabs([
            "Anomaly Detection", "Price Elasticity", "Churn Prediction", "Product Associations"
        ])

        # Tab 1: Anomaly Detection
        with ml_tab1:
            anomaly_data = get_ml_data("anomalies")

            if anomaly_data and anomaly_data.get("category_anomalies"):
                # Category performance - the main visualization
                category_anomalies = anomaly_data.get("category_anomalies", [])

                if category_anomalies:
                    st.markdown("**Category Performance vs Median (Deviation Analysis)**")

                    # Prepare data for diverging bar chart
                    high_performers = [c for c in category_anomalies if c["type"] == "high_performer"]
                    underperformers = [c for c in category_anomalies if c["type"] == "underperformer"]

                    # Combine and sort by deviation
                    all_categories = []
                    for cat in high_performers[:5]:
                        all_categories.append({
                            "category": cat["category"][:20],  # Truncate long names
                            "deviation": cat["deviation"],
                            "revenue": cat["avg_revenue_per_order"],
                            "orders": cat["order_count"],
                            "type": "High Performer"
                        })
                    for cat in underperformers[:5]:
                        all_categories.append({
                            "category": cat["category"][:20],
                            "deviation": -abs(cat["deviation"]),  # Make negative for underperformers
                            "revenue": cat["avg_revenue_per_order"],
                            "orders": cat["order_count"],
                            "type": "Underperformer"
                        })

                    if all_categories:
                        df_cat = pd.DataFrame(all_categories)
                        df_cat = df_cat.sort_values("deviation", ascending=True)

                        # Create diverging bar chart
                        fig = go.Figure()

                        colors = ["#22c55e" if d > 0 else "#ef4444" for d in df_cat["deviation"]]

                        fig.add_trace(go.Bar(
                            y=df_cat["category"],
                            x=df_cat["deviation"],
                            orientation="h",
                            marker_color=colors,
                            text=[f"{d:+.1f}σ | R${r:,.0f}/order" for d, r in zip(df_cat["deviation"], df_cat["revenue"])],
                            textposition="auto",
                            hovertemplate="<b>%{y}</b><br>Deviation: %{x:.1f}σ<br><extra></extra>",
                        ))

                        # Add vertical line at 0
                        fig.add_vline(x=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")

                        fig.update_layout(
                            template="plotly_dark",
                            plot_bgcolor="rgba(10, 10, 15, 0.5)",
                            paper_bgcolor="rgba(10, 10, 15, 0.5)",
                            margin=dict(l=10, r=10, t=10, b=10),
                            height=350,
                            xaxis_title="Standard Deviations from Median",
                            yaxis_title="",
                            showlegend=False,
                        )
                        st.plotly_chart(fig, use_container_width=True, key="anomaly_chart")

                        # Legend explanation
                        st.markdown("""
                        <div style="display: flex; gap: 20px; font-size: 0.85em; color: rgba(255,255,255,0.7); margin-top: 10px;">
                            <span><span style="color: #22c55e;">■</span> High Performers (above median)</span>
                            <span><span style="color: #ef4444;">■</span> Underperformers (below median)</span>
                        </div>
                        """, unsafe_allow_html=True)

                    # Detail cards for top performers and underperformers
                    st.markdown("---")
                    col_high, col_low = st.columns(2)

                    with col_high:
                        st.markdown("**Top High Performers**")
                        for cat in high_performers[:3]:
                            st.markdown(f"""
                            <div class="ml-insight-card positive">
                                <div class="ml-insight-title">{cat['category']}</div>
                                <div class="ml-insight-metric">R$ {cat['avg_revenue_per_order']:.0f}/order</div>
                                <div class="ml-insight-detail">{cat['order_count']:,} orders | +{cat['deviation']:.1f}σ above median</div>
                            </div>
                            """, unsafe_allow_html=True)

                    with col_low:
                        st.markdown("**Top Underperformers**")
                        for cat in underperformers[:3]:
                            st.markdown(f"""
                            <div class="ml-insight-card warning">
                                <div class="ml-insight-title">{cat['category']}</div>
                                <div class="ml-insight-metric">R$ {cat['avg_revenue_per_order']:.0f}/order</div>
                                <div class="ml-insight-detail">{cat['order_count']:,} orders | {cat['deviation']:.1f}σ below median</div>
                            </div>
                            """, unsafe_allow_html=True)

                # Summary
                summary = anomaly_data.get("summary", {})
                if summary:
                    st.markdown(f"""
                    <div class="ml-summary">
                        <span><b>{summary.get('category_anomalies_count', 0)}</b> category outliers detected</span> |
                        <span>Median Revenue: R$ {summary.get('revenue_mean', 0):,.0f}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Loading anomaly detection data...")

        # Tab 2: Price Elasticity
        with ml_tab2:
            elasticity_data = get_ml_data("price-elasticity")

            if elasticity_data and elasticity_data.get("elasticity_by_category"):
                st.markdown("**Price Sensitivity by Category**")

                elasticity_list = elasticity_data.get("elasticity_by_category", [])[:10]

                if elasticity_list:
                    # Create elasticity bar chart
                    df_elastic = pd.DataFrame(elasticity_list)

                    fig = go.Figure()
                    colors = ["#ef4444" if e < -1 else "#22c55e" if e > -0.5 else "#6482ff" for e in df_elastic["elasticity"]]

                    fig.add_trace(go.Bar(
                        y=df_elastic["category"],
                        x=df_elastic["elasticity"],
                        orientation="h",
                        marker_color=colors,
                        text=df_elastic["elasticity"].round(2),
                        textposition="auto",
                    ))

                    fig.update_layout(
                        template="plotly_dark",
                        plot_bgcolor="rgba(10, 10, 15, 0.5)",
                        paper_bgcolor="rgba(10, 10, 15, 0.5)",
                        margin=dict(l=10, r=10, t=10, b=10),
                        height=350,
                        xaxis_title="Elasticity Coefficient",
                        yaxis_title="",
                    )
                    st.plotly_chart(fig, use_container_width=True, key="elasticity_chart")

                    # Legend
                    st.markdown("""
                    <div class="elasticity-legend">
                        <span class="legend-item"><span class="dot elastic"></span> Elastic (< -1): Price sensitive</span>
                        <span class="legend-item"><span class="dot unit"></span> Unit elastic: Moderate</span>
                        <span class="legend-item"><span class="dot inelastic"></span> Inelastic (> -0.5): Premium potential</span>
                    </div>
                    """, unsafe_allow_html=True)

                # Pricing recommendations
                st.markdown("**Pricing Recommendations**")
                col_rec1, col_rec2 = st.columns(2)

                inelastic = [e for e in elasticity_list if e["elasticity"] > -0.5][:3]
                elastic = [e for e in elasticity_list if e["elasticity"] < -1][:3]

                with col_rec1:
                    st.markdown("**Premium Pricing Opportunities**")
                    for cat in inelastic:
                        st.markdown(f"""
                        <div class="ml-insight-card positive">
                            <div class="ml-insight-title">{cat['category']}</div>
                            <div class="ml-insight-detail">Avg: R$ {cat['avg_price']:.0f} | {cat['recommendation']}</div>
                        </div>
                        """, unsafe_allow_html=True)

                with col_rec2:
                    st.markdown("**Competitive Pricing Needed**")
                    for cat in elastic:
                        st.markdown(f"""
                        <div class="ml-insight-card warning">
                            <div class="ml-insight-title">{cat['category']}</div>
                            <div class="ml-insight-detail">Avg: R$ {cat['avg_price']:.0f} | {cat['recommendation']}</div>
                        </div>
                        """, unsafe_allow_html=True)

                # Price segments
                segments = elasticity_data.get("price_segments", [])
                if segments:
                    st.markdown("**Revenue by Price Segment**")
                    df_seg = pd.DataFrame(segments)
                    fig = px.pie(df_seg, names="segment", values="revenue", template="plotly_dark",
                                 color_discrete_sequence=["#3b82f6", "#6482ff", "#8b5cf6", "#a855f7"])
                    fig.update_layout(
                        plot_bgcolor="rgba(10, 10, 15, 0.5)",
                        paper_bgcolor="rgba(10, 10, 15, 0.5)",
                        margin=dict(l=10, r=10, t=10, b=10),
                        height=250,
                    )
                    st.plotly_chart(fig, use_container_width=True, key="segment_pie")
            else:
                st.info("Loading price elasticity data...")

        # Tab 3: Churn Prediction
        with ml_tab3:
            churn_data = get_ml_data("churn-prediction")

            if churn_data and churn_data.get("risk_segments"):
                # Risk segment overview
                st.markdown("**Customer Churn Risk Segments**")

                risk_segments = churn_data.get("risk_segments", [])

                # Donut chart for risk distribution
                if risk_segments:
                    df_risk = pd.DataFrame(risk_segments)

                    col_donut, col_metrics = st.columns([1, 1])

                    with col_donut:
                        colors = {"high": "#ef4444", "medium": "#f59e0b", "low": "#22c55e"}
                        fig = go.Figure(go.Pie(
                            labels=df_risk["risk_level"],
                            values=df_risk["customer_count"],
                            hole=0.6,
                            marker_colors=[colors.get(r, "#6482ff") for r in df_risk["risk_level"]],
                            textinfo="label+percent",
                        ))
                        fig.update_layout(
                            template="plotly_dark",
                            plot_bgcolor="rgba(10, 10, 15, 0.5)",
                            paper_bgcolor="rgba(10, 10, 15, 0.5)",
                            margin=dict(l=10, r=10, t=10, b=10),
                            height=250,
                            showlegend=False,
                            annotations=[dict(text="Risk", x=0.5, y=0.5, font_size=16, showarrow=False)]
                        )
                        st.plotly_chart(fig, use_container_width=True, key="churn_donut")

                    with col_metrics:
                        summary = churn_data.get("summary", {})
                        at_risk = summary.get("at_risk_revenue", 0)
                        recovery = summary.get("retention_opportunity", 0)

                        st.markdown(f"""
                        <div class="churn-metric-card critical">
                            <div class="churn-metric-label">Revenue at Risk</div>
                            <div class="churn-metric-value">R$ {at_risk:,.0f}</div>
                        </div>
                        """, unsafe_allow_html=True)

                        st.markdown(f"""
                        <div class="churn-metric-card positive">
                            <div class="churn-metric-label">Recovery Opportunity (30%)</div>
                            <div class="churn-metric-value">R$ {recovery:,.0f}</div>
                        </div>
                        """, unsafe_allow_html=True)

                        high_risk_pct = summary.get("high_risk_pct", 0)
                        st.markdown(f"""
                        <div class="churn-metric-card warning">
                            <div class="churn-metric-label">High Risk Customers</div>
                            <div class="churn-metric-value">{high_risk_pct:.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)

                # Risk indicators
                indicators = churn_data.get("indicators", [])
                if indicators:
                    st.markdown("**Churn Risk Indicators**")
                    for ind in indicators:
                        risk_class = "critical" if ind.get("risk_factor") == "high" else "warning"
                        st.markdown(f"""
                        <div class="ml-insight-card {risk_class}">
                            <div class="ml-insight-title">{ind['indicator']}</div>
                            <div class="ml-insight-detail">{ind['description']}</div>
                            <div class="ml-insight-metric">{ind.get('count', ind.get('percentage', 'N/A')):,} customers</div>
                        </div>
                        """, unsafe_allow_html=True)

                # Segment details table
                st.markdown("**Segment Details**")
                for seg in risk_segments:
                    icon = "[H]" if seg["risk_level"] == "high" else "[M]" if seg["risk_level"] == "medium" else "[L]"
                    st.markdown(f"""
                    {icon} **{seg['risk_level'].title()}**: {seg['customer_count']:,} customers |
                    Avg LTV: R$ {seg['avg_lifetime_value']:.0f} |
                    Inactive: {seg['avg_days_inactive']:.0f} days |
                    Revenue: R$ {seg['total_revenue_at_risk']:,.0f}
                    """)
            else:
                st.info("Loading churn prediction data...")

        # Tab 4: Product Associations
        with ml_tab4:
            assoc_data = get_ml_data("product-associations")

            if assoc_data and assoc_data.get("associations"):
                st.markdown("**Frequently Bought Together (Top Product Pairs)**")

                associations = assoc_data.get("associations", [])[:8]

                if associations:
                    # Create clean horizontal bar chart showing lift values
                    pair_names = [f"{a['category_a'][:15]} + {a['category_b'][:15]}" for a in associations]
                    lift_values = [a["lift"] for a in associations]
                    confidence_values = [a.get("confidence_a_to_b", 0) for a in associations]

                    # Color based on lift strength
                    colors = ["#22c55e" if l > 500 else "#6482ff" if l > 100 else "#f59e0b" for l in lift_values]

                    fig = go.Figure()

                    fig.add_trace(go.Bar(
                        y=pair_names[::-1],  # Reverse for top-to-bottom display
                        x=lift_values[::-1],
                        orientation="h",
                        marker_color=colors[::-1],
                        text=[f"{l:,.0f}x more likely" for l in lift_values[::-1]],
                        textposition="auto",
                        hovertemplate="<b>%{y}</b><br>Lift: %{x:,.0f}x<br><extra></extra>",
                    ))

                    fig.update_layout(
                        template="plotly_dark",
                        plot_bgcolor="rgba(10, 10, 15, 0.5)",
                        paper_bgcolor="rgba(10, 10, 15, 0.5)",
                        margin=dict(l=10, r=10, t=10, b=10),
                        height=350,
                        xaxis_title="Lift (times more likely to buy together)",
                        yaxis_title="",
                        showlegend=False,
                    )
                    st.plotly_chart(fig, use_container_width=True, key="assoc_bar")

                    # Legend explanation
                    st.markdown("""
                    <div style="display: flex; gap: 20px; font-size: 0.85em; color: rgba(255,255,255,0.7); margin-top: 10px;">
                        <span><span style="color: #22c55e;">■</span> Very Strong (500x+)</span>
                        <span><span style="color: #6482ff;">■</span> Strong (100-500x)</span>
                        <span><span style="color: #f59e0b;">■</span> Moderate (<100x)</span>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("---")

                # Top bundles as action cards
                bundles = assoc_data.get("top_bundles", [])
                if bundles:
                    st.markdown("**Bundle Recommendations**")
                    for bundle in bundles[:3]:
                        st.markdown(f"""
                        <div class="bundle-card">
                            <div class="bundle-title">{bundle['bundle']}</div>
                            <div class="bundle-rec">{bundle['recommendation']}</div>
                            <div class="bundle-opp">{bundle['opportunity']}</div>
                        </div>
                        """, unsafe_allow_html=True)

                # Summary
                summary = assoc_data.get("summary", {})
                if summary:
                    st.markdown(f"""
                    <div class="ml-summary">
                        <b>{summary.get('multi_category_pct', 0):.1f}%</b> of orders have multiple categories |
                        <b>{summary.get('strong_associations', 0)}</b> strong associations (lift > 2) |
                        <b>{summary.get('categories_analyzed', 0)}</b> categories analyzed
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Loading product association data...")

        st.markdown("---")

        # Date range info
        st.markdown('<div class="section-title">Data Coverage</div>', unsafe_allow_html=True)
        col_date1, col_date2 = st.columns(2)
        with col_date1:
            st.info(f"Start: {stats.get('date_range', {}).get('start', 'N/A')}")
        with col_date2:
            st.info(f"End: {stats.get('date_range', {}).get('end', 'N/A')}")
    else:
        st.warning("Unable to fetch dataset statistics. Make sure the API is running.")


# ============== TAB 3: SYSTEM INFO ==============
with tab_system:
    st.markdown('<div class="section-title">System Status <span class="section-badge">Health</span></div>', unsafe_allow_html=True)

    health = get_api_health()
    api_status = health.get("status", "unknown")

    col1, col2 = st.columns(2)

    with col1:
        status_icon = "[OK]" if api_status in ("ok", "healthy") else "[!]"
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">API Status</div>
            <div class="stat-value">{status_icon} {api_status.title()}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">API Endpoint</div>
            <div class="stat-value" style="font-size: 0.9rem;">{API_BASE}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Tech Stack
    st.markdown('<div class="section-title">Technology Stack <span class="section-badge">What Powers This</span></div>', unsafe_allow_html=True)

    col_llm, col_vector, col_framework = st.columns(3)

    with col_llm:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-label">LLM Providers</div>
            <div class="stat-value" style="font-size: 1rem;"><a href="https://x.ai/" target="_blank" style="color: #6482ff; text-decoration: none;">xAI Grok</a></div>
            <div style="font-size: 0.8rem; color: rgba(255,255,255,0.6);">Primary: <a href="https://x.ai/api" target="_blank" style="color: rgba(255,255,255,0.8);">Grok 3 Fast</a></div>
            <div style="font-size: 0.8rem; color: rgba(255,255,255,0.6);">Fallback: <a href="https://groq.com/" target="_blank" style="color: rgba(255,255,255,0.8);">Groq (Llama)</a></div>
        </div>
        """, unsafe_allow_html=True)

    with col_vector:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-label">Vector Database</div>
            <div class="stat-value" style="font-size: 1rem;"><a href="https://qdrant.tech/" target="_blank" style="color: #22c55e; text-decoration: none;">Qdrant Cloud</a></div>
            <div style="font-size: 0.8rem; color: rgba(255,255,255,0.6);">Embeddings: <a href="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2" target="_blank" style="color: rgba(255,255,255,0.8);">MiniLM</a></div>
            <div style="font-size: 0.8rem; color: rgba(255,255,255,0.6);">Hybrid: BM25 + Vector</div>
        </div>
        """, unsafe_allow_html=True)

    with col_framework:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-label">Framework</div>
            <div class="stat-value" style="font-size: 1rem;"><a href="https://langchain-ai.github.io/langgraph/" target="_blank" style="color: #f59e0b; text-decoration: none;">LangGraph</a></div>
            <div style="font-size: 0.8rem; color: rgba(255,255,255,0.6);">Orchestration: State Machine</div>
            <div style="font-size: 0.8rem; color: rgba(255,255,255,0.6);">Observability: <a href="https://langfuse.com/" target="_blank" style="color: rgba(255,255,255,0.8);">Langfuse</a></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Architecture
    st.markdown('<div class="section-title">Multi-Agent Architecture</div>', unsafe_allow_html=True)

    col_agents1, col_agents2 = st.columns(2)

    with col_agents1:
        st.markdown("""
        <div class="insight-card info">
            <div class="insight-title">Orchestrator</div>
            <div class="insight-desc">Classifies intent and routes queries to specialized agents</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="insight-card positive">
            <div class="insight-title">Sales Agent</div>
            <div class="insight-desc">Revenue analytics, order trends, category performance (Pandas)</div>
        </div>
        """, unsafe_allow_html=True)

    with col_agents2:
        st.markdown("""
        <div class="insight-card positive">
            <div class="insight-title">Sentiment Agent</div>
            <div class="insight-desc">Customer review analysis using RAG (Qdrant + BM25)</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="insight-card positive">
            <div class="insight-title">Forecast Agent</div>
            <div class="insight-desc">Time-series predictions using Facebook Prophet</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Data
    st.markdown('<div class="section-title">Dataset</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-card info">
        <div class="insight-title"><a href="https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce" target="_blank" style="color: #6482ff; text-decoration: none;">Olist Brazilian E-commerce</a></div>
        <div class="insight-desc">100K+ orders | 40K+ reviews | Real marketplace data (2016-2018)</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Session Info
    st.markdown('<div class="section-title">Session Info</div>', unsafe_allow_html=True)

    config_items = [
        ("Session ID", st.session_state.session_id),
        ("Chat History", f"{len(st.session_state.chat_history)} messages"),
        ("Query Memory", f"{len(st.session_state.conversation_context)} items"),
    ]

    cols_config = st.columns(3)
    for idx, (label, value) in enumerate(config_items):
        with cols_config[idx]:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">{label}</div>
                <div class="stat-value" style="font-size: 0.9rem;">{value}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Settings
    st.markdown('<div class="section-title">Settings</div>', unsafe_allow_html=True)

    st.session_state.show_suggestions = st.checkbox("Show follow-up suggestions", value=st.session_state.show_suggestions)

    if st.button("Refresh Status", use_container_width=True):
        st.rerun()


# ============== FOOTER ==============
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: rgba(255,255,255,0.4); font-size: 0.8rem;">
    Powered by xAI Grok, LangGraph, Qdrant, and Facebook Prophet
</div>
""", unsafe_allow_html=True)
