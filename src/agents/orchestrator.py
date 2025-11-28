"""LangGraph orchestrator tying all agents together."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from langgraph.graph import END, StateGraph

from src.agents.forecast_agent import ForecastAgent
from src.agents.sales_agent import SalesAgent
from src.agents.sentiment_agent import SentimentAgent
from src.agents.state import AgentState, Intent
from src.config.settings import get_settings
from src.llm.clients import call_xai_chat

# Agent execution timeout in seconds
AGENT_TIMEOUT_SECONDS = 30


class Orchestrator:
    def __init__(
        self,
        sales_agent: SalesAgent | None = None,
        sentiment_agent: SentimentAgent | None = None,
        forecast_agent: ForecastAgent | None = None,
        timeout_seconds: int = AGENT_TIMEOUT_SECONDS,
    ):
        self.sales_agent = sales_agent or SalesAgent()
        self.sentiment_agent = sentiment_agent or SentimentAgent()
        self.forecast_agent = forecast_agent or ForecastAgent()
        self.timeout_seconds = timeout_seconds
        self.graph = self._build_graph()

    def _run_with_timeout(self, func, state: AgentState, agent_name: str) -> AgentState:
        """Run an agent function with a timeout."""
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, state)
            try:
                return future.result(timeout=self.timeout_seconds)
            except FuturesTimeoutError:
                # Return state with timeout error info
                state[f"{agent_name}_result"] = {
                    "error": f"{agent_name} agent timed out after {self.timeout_seconds}s",
                    "timeout": True,
                }
                agents_used = state.get("agents_used", [])
                agents_used.append(f"{agent_name}_timeout")
                state["agents_used"] = agents_used
                return state
            except Exception as e:
                # Return state with error info
                state[f"{agent_name}_result"] = {
                    "error": f"{agent_name} agent error: {str(e)}",
                }
                agents_used = state.get("agents_used", [])
                agents_used.append(f"{agent_name}_error")
                state["agents_used"] = agents_used
                return state

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("classify_intent", self.classify_intent)
        workflow.add_node("sales_agent", self._sales_with_timeout)
        workflow.add_node("sentiment_agent", self._sentiment_with_timeout)
        workflow.add_node("forecast_agent", self._forecast_with_timeout)
        workflow.add_node("multi_agent", self._run_multi)
        workflow.add_node("synthesize", self.synthesize_response)

        workflow.set_entry_point("classify_intent")
        workflow.add_conditional_edges(
            "classify_intent",
            self.route_by_intent,
            {
                "sales": "sales_agent",
                "sentiment": "sentiment_agent",
                "forecast": "forecast_agent",
                "multi": "multi_agent",
            },
        )

        workflow.add_edge("sales_agent", "synthesize")
        workflow.add_edge("sentiment_agent", "synthesize")
        workflow.add_edge("forecast_agent", "synthesize")
        workflow.add_edge("multi_agent", "synthesize")
        workflow.add_edge("synthesize", END)
        return workflow.compile()

    def _sales_with_timeout(self, state: AgentState) -> AgentState:
        """Run sales agent with timeout."""
        return self._run_with_timeout(self.sales_agent.invoke, state, "sales")

    def _sentiment_with_timeout(self, state: AgentState) -> AgentState:
        """Run sentiment agent with timeout."""
        return self._run_with_timeout(self.sentiment_agent.invoke, state, "sentiment")

    def _forecast_with_timeout(self, state: AgentState) -> AgentState:
        """Run forecast agent with timeout."""
        return self._run_with_timeout(self.forecast_agent.invoke, state, "forecast")

    def _run_multi(self, state: AgentState) -> AgentState:
        """Run sales + sentiment agents in parallel for multi-intent queries."""
        import copy

        # Create copies of state for parallel execution
        sales_state = copy.deepcopy(state)
        sentiment_state = copy.deepcopy(state)

        with ThreadPoolExecutor(max_workers=2) as executor:
            sales_future = executor.submit(self.sales_agent.invoke, sales_state)
            sentiment_future = executor.submit(self.sentiment_agent.invoke, sentiment_state)

            # Wait for both with timeout
            try:
                sales_result_state = sales_future.result(timeout=self.timeout_seconds)
                state["sales_result"] = sales_result_state.get("sales_result")
            except (FuturesTimeoutError, Exception) as e:
                state["sales_result"] = {"error": f"Sales agent failed: {str(e)}"}

            try:
                sentiment_result_state = sentiment_future.result(timeout=self.timeout_seconds)
                state["sentiment_result"] = sentiment_result_state.get("sentiment_result")
            except (FuturesTimeoutError, Exception) as e:
                state["sentiment_result"] = {"error": f"Sentiment agent failed: {str(e)}"}

        # Merge agents_used from both
        agents_used = state.get("agents_used", [])
        if state.get("sales_result") and "error" not in state.get("sales_result", {}):
            agents_used.append("sales")
        if state.get("sentiment_result") and "error" not in state.get("sentiment_result", {}):
            agents_used.append("sentiment")
        state["agents_used"] = agents_used

        return state

    def classify_intent(self, state: AgentState) -> AgentState:
        question = state["query"]
        settings = get_settings()

        # FAST PATH: Use keyword-based classification first (instant, no API call)
        intent = self._fast_keyword_classify(question)

        # If keyword classification is confident, use it
        if intent != "unknown":
            state["intent"] = intent
            state["messages"] = state.get("messages", [])
            return state

        # Try fast classification with Bedrock Claude Haiku if enabled
        if settings.use_bedrock_for_intent:
            try:
                from src.aws.bedrock import get_bedrock_client
                bedrock = get_bedrock_client()
                if bedrock and bedrock.enabled:
                    intent = bedrock.classify_intent_fast(question)
                    state["intent"] = intent
                    state["messages"] = state.get("messages", [])
                    return state
            except Exception:
                pass  # Fall back to xAI

        # Use xAI for intent classification (only if keyword classification failed)
        messages = [
            {
                "role": "system",
                "content": (
                    "Classify marketing analytics questions into EXACTLY one of these categories:\n"
                    "- 'sales': revenue, products, orders, categories, growth, performance\n"
                    "- 'sentiment': reviews, complaints, feedback, ratings, customer opinions\n"
                    "- 'forecast': predictions, next month, future, trends, projections\n"
                    "- 'multi': questions needing both sales data AND sentiment analysis\n\n"
                    "Respond with ONLY valid JSON: {\"intent\": \"sales|sentiment|forecast|multi\"}"
                ),
            },
            {"role": "user", "content": question},
        ]
        raw, usage = call_xai_chat(messages, max_tokens=100, return_usage=True)

        # Track token usage from intent classification
        state["input_tokens"] = state.get("input_tokens", 0) + usage.get("input_tokens", 0)
        state["output_tokens"] = state.get("output_tokens", 0) + usage.get("output_tokens", 0)

        # Parse and validate intent
        valid_intents = {"sales", "sentiment", "forecast", "multi"}
        intent: Intent = "sales"  # default fallback

        try:
            # Try to parse JSON
            parsed = json.loads(raw)
            raw_intent = parsed.get("intent", "").lower().strip()
            if raw_intent in valid_intents:
                intent = raw_intent
            else:
                # Map common variations
                intent = self._map_intent(raw_intent, question)
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract intent from raw text
            raw_lower = raw.lower()
            for valid in valid_intents:
                if valid in raw_lower:
                    intent = valid
                    break
            else:
                intent = self._map_intent("", question)

        state["intent"] = intent
        state["messages"] = state.get("messages", [])
        return state

    def _fast_keyword_classify(self, question: str) -> Intent | str:
        """Ultra-fast keyword-based intent classification. Returns 'unknown' if unsure."""
        q = question.lower()

        # Strong forecast indicators
        forecast_strong = ["forecast", "predict", "next month", "next quarter", "future", "projection", "will be", "expect"]
        if any(kw in q for kw in forecast_strong):
            return "forecast"

        # Strong sentiment indicators
        sentiment_strong = ["review", "complaint", "feedback", "rating", "sentiment", "opinion", "customer say", "customers think", "satisfaction"]
        if any(kw in q for kw in sentiment_strong):
            return "sentiment"

        # Check for multi (both sales and sentiment keywords)
        has_sales = any(kw in q for kw in ["revenue", "sales", "sell", "product", "order", "category", "top", "best", "performance"])
        has_sentiment = any(kw in q for kw in sentiment_strong)

        if has_sales and has_sentiment:
            return "multi"

        # Strong sales indicators
        sales_strong = ["revenue", "sales", "selling", "product", "order", "category", "top", "best", "performance", "growth", "compare", "region", "state", "month", "trend", "driving"]
        if any(kw in q for kw in sales_strong):
            return "sales"

        # Default to sales for general questions (most common case)
        return "sales"

    def _map_intent(self, raw_intent: str, question: str) -> Intent:
        """Map unknown intents to valid ones based on keywords."""
        question_lower = question.lower()

        # Check for forecast keywords
        forecast_keywords = ["forecast", "predict", "next month", "future", "projection", "trend"]
        if any(kw in question_lower for kw in forecast_keywords):
            return "forecast"

        # Check for sentiment keywords
        sentiment_keywords = ["complaint", "review", "feedback", "rating", "sentiment", "opinion", "customer say"]
        if any(kw in question_lower for kw in sentiment_keywords):
            return "sentiment"

        # Check for multi keywords (needs both)
        if ("revenue" in question_lower or "sales" in question_lower) and \
           any(kw in question_lower for kw in sentiment_keywords):
            return "multi"

        # Default to sales
        return "sales"

    def route_by_intent(self, state: AgentState) -> Intent:
        return state["intent"]

    def synthesize_response(self, state: AgentState) -> AgentState:
        prompt = self._build_synthesis_prompt(state)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a marketing executive consultant. "
                    "Combine the structured data into a crisp narrative with a recommendation."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        answer, usage = call_xai_chat(messages, max_tokens=800, return_usage=True)
        state["final_answer"] = answer
        state["sources"] = self._collect_sources(state)

        # Track token usage
        state["input_tokens"] = state.get("input_tokens", 0) + usage.get("input_tokens", 0)
        state["output_tokens"] = state.get("output_tokens", 0) + usage.get("output_tokens", 0)

        return state

    def _build_synthesis_prompt(self, state: AgentState) -> str:
        return json.dumps(
            {
                "question": state["query"],
                "sales": state.get("sales_result"),
                "sentiment": state.get("sentiment_result"),
                "forecast": state.get("forecast_result"),
            },
            indent=2,
        )

    def _collect_sources(self, state: AgentState) -> list[dict]:
        sentiment = state.get("sentiment_result") or {}
        return sentiment.get("sources", [])

    def invoke(self, query: str) -> AgentState:
        initial_state: AgentState = {"query": query, "messages": []}
        return self.graph.invoke(initial_state)

