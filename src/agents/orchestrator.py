"""LangGraph orchestrator tying all agents together."""

from __future__ import annotations

import json

from langgraph.graph import END, StateGraph

from src.agents.forecast_agent import ForecastAgent
from src.agents.sales_agent import SalesAgent
from src.agents.sentiment_agent import SentimentAgent
from src.agents.state import AgentState, Intent
from src.llm.clients import call_xai_chat


class Orchestrator:
    def __init__(
        self,
        sales_agent: SalesAgent | None = None,
        sentiment_agent: SentimentAgent | None = None,
        forecast_agent: ForecastAgent | None = None,
    ):
        self.sales_agent = sales_agent or SalesAgent()
        self.sentiment_agent = sentiment_agent or SentimentAgent()
        self.forecast_agent = forecast_agent or ForecastAgent()
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("classify_intent", self.classify_intent)
        workflow.add_node("sales_agent", self.sales_agent.invoke)
        workflow.add_node("sentiment_agent", self.sentiment_agent.invoke)
        workflow.add_node("forecast_agent", self.forecast_agent.invoke)
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

    def _run_multi(self, state: AgentState) -> AgentState:
        """Run sales + sentiment agents sequentially for multi-intent queries."""
        state = self.sales_agent.invoke(state)
        state = self.sentiment_agent.invoke(state)
        return state

    def classify_intent(self, state: AgentState) -> AgentState:
        question = state["query"]
        messages = [
            {
                "role": "system",
                "content": (
                    "Classify marketing analytics questions. "
                    "Label as 'sales', 'sentiment', 'forecast', or 'multi'. "
                    "Respond with JSON: {\"intent\": \"...\", \"reason\": \"...\"}."
                ),
            },
            {"role": "user", "content": question},
        ]
        raw = call_xai_chat(messages, max_tokens=100)
        try:
            parsed = json.loads(raw)
            intent: Intent = parsed.get("intent", "sales")
        except json.JSONDecodeError:
            intent = "sales"
        state["intent"] = intent
        state["messages"] = state.get("messages", [])
        return state

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
        answer = call_xai_chat(messages, max_tokens=800)
        state["final_answer"] = answer
        state["sources"] = self._collect_sources(state)
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

