"""Sales analysis agent that runs SQL-like aggregations via pandas."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config.settings import get_settings
from src.agents.state import AgentState


@dataclass
class SalesAgent:
    orders_view_path: Path | None = None

    def __post_init__(self) -> None:
        settings = get_settings()
        self.orders_view_path = self.orders_view_path or Path(settings.data_processed_dir) / "orders_view.parquet"
        if not self.orders_view_path.exists():
            raise FileNotFoundError(
                f"{self.orders_view_path} not found. Run src/data/etl.py to generate processed views."
            )
        self.df = pd.read_parquet(self.orders_view_path)
        self.df["order_purchase_month"] = (
            self.df["order_purchase_timestamp"].dt.to_period("M").dt.to_timestamp()
        )

    def invoke(self, state: AgentState) -> AgentState:
        """Entry point for LangGraph node."""

        sales_payload = self.revenue_by_category(limit=5)
        state["sales_result"] = sales_payload
        agents_used = state.get("agents_used", [])
        agents_used.append("sales")
        state["agents_used"] = agents_used
        return state

    def revenue_by_category(self, limit: int = 5) -> dict:
        revenue = (
            self.df.groupby(["product_category_name_english"])
            .agg(total_revenue=("price", "sum"), avg_ticket=("price", "mean"), orders=("order_id", "nunique"))
            .reset_index()
            .sort_values("total_revenue", ascending=False)
            .head(limit)
        )
        revenue["total_revenue"] = revenue["total_revenue"].round(2)
        revenue["avg_ticket"] = revenue["avg_ticket"].round(2)
        return {
            "summary": revenue.to_dict(orient="records"),
            "currency": "BRL",
        }

