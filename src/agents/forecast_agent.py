"""Forecast agent using a simple moving-average baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.agents.state import AgentState
from src.config.settings import get_settings


@dataclass
class ForecastAgent:
    orders_view_path: Path | None = None
    window_months: int = 3

    def __post_init__(self) -> None:
        settings = get_settings()
        self.orders_view_path = self.orders_view_path or Path(settings.data_processed_dir) / "orders_view.parquet"
        if not self.orders_view_path.exists():
            raise FileNotFoundError(
                f"{self.orders_view_path} not found. Run src/data/etl.py to generate processed views."
            )
        df = pd.read_parquet(self.orders_view_path)
        df["order_purchase_month"] = (
            df["order_purchase_timestamp"].dt.to_period("M").dt.to_timestamp()
        )
        self.monthly = (
            df.groupby("order_purchase_month")
            .agg(total_revenue=("price", "sum"))
            .sort_index()
        )

    def invoke(self, state: AgentState) -> AgentState:
        forecast = self._moving_average_forecast()
        state["forecast_result"] = forecast
        agents_used = state.get("agents_used", [])
        agents_used.append("forecast")
        state["agents_used"] = agents_used
        return state

    def _moving_average_forecast(self) -> dict:
        tail = self.monthly.tail(self.window_months)
        prediction = tail["total_revenue"].mean()
        last_period = tail.index.max()
        next_period = (last_period.to_period("M") + 1).to_timestamp()
        return {
            "method": "moving_average",
            "window_months": self.window_months,
            "history": tail.reset_index().to_dict(orient="records"),
            "forecast": {
                "period_start": next_period.isoformat(),
                "total_revenue": round(float(prediction), 2),
                "currency": "BRL",
            },
        }

