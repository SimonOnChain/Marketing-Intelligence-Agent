"""Shared LangGraph state definition."""

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


Intent = Literal["sales", "sentiment", "forecast", "multi"]


class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    query: str
    intent: Intent
    sales_result: dict | None
    sentiment_result: dict | None
    forecast_result: dict | None
    final_answer: str | None
    sources: list[dict] | None
    agents_used: list[str]
    input_tokens: int
    output_tokens: int

