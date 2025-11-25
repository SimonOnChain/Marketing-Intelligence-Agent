"""Sentiment analysis agent built on the RAG chain."""

from dataclasses import dataclass

from src.agents.state import AgentState
from src.retrieval.rag_chain import RAGChain, build_rag_chain


@dataclass
class SentimentAgent:
    chain: RAGChain | None = None

    def __post_init__(self) -> None:
        self.chain = self.chain or build_rag_chain()

    def invoke(self, state: AgentState) -> AgentState:
        query = state["query"]
        prompt = (
            f"Summarize customer sentiment relevant to the question: {query}. "
            "Highlight most positive and negative themes with supporting review IDs."
        )
        rag_result = self.chain.invoke(prompt)
        state["sentiment_result"] = rag_result
        agents_used = state.get("agents_used", [])
        agents_used.append("sentiment")
        state["agents_used"] = agents_used
        return state

