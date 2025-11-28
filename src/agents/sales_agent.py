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
        query = state.get("query", "").lower()

        # Determine which analysis to run based on query
        if "month" in query or "trend" in query or "time" in query:
            sales_payload = self.revenue_by_month()
        elif "state" in query or "region" in query or "geographic" in query:
            sales_payload = self.revenue_by_state()
        else:
            sales_payload = self.revenue_by_category(limit=10)

        state["sales_result"] = sales_payload
        agents_used = state.get("agents_used", [])
        agents_used.append("sales")
        state["agents_used"] = agents_used
        return state

    def revenue_by_category(self, limit: int = 10) -> dict:
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
            "chart_type": "bar",
            "chart_title": "Revenue by Category",
            "x_field": "product_category_name_english",
            "y_field": "total_revenue",
            "summary": revenue.to_dict(orient="records"),
            "currency": "BRL",
        }

    def revenue_by_month(self) -> dict:
        """Get monthly revenue trend."""
        monthly = (
            self.df.groupby(self.df["order_purchase_month"].dt.strftime("%Y-%m"))
            .agg(total_revenue=("price", "sum"), orders=("order_id", "nunique"))
            .reset_index()
            .rename(columns={"order_purchase_month": "month"})
            .sort_values("month")
        )
        monthly["total_revenue"] = monthly["total_revenue"].round(2)
        return {
            "chart_type": "line",
            "chart_title": "Monthly Revenue Trend",
            "x_field": "month",
            "y_field": "total_revenue",
            "summary": monthly.to_dict(orient="records"),
            "currency": "BRL",
        }

    def revenue_by_state(self) -> dict:
        """Get revenue by customer state."""
        by_state = (
            self.df.groupby("customer_state")
            .agg(total_revenue=("price", "sum"), orders=("order_id", "nunique"))
            .reset_index()
            .sort_values("total_revenue", ascending=False)
            .head(10)
        )
        by_state["total_revenue"] = by_state["total_revenue"].round(2)
        return {
            "chart_type": "bar",
            "chart_title": "Revenue by State",
            "x_field": "customer_state",
            "y_field": "total_revenue",
            "summary": by_state.to_dict(orient="records"),
            "currency": "BRL",
        }

