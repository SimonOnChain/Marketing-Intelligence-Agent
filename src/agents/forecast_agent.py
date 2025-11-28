"""Forecast agent using Amazon Forecast with local fallback."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.agents.state import AgentState
from src.config.settings import get_settings
from src.aws.forecast import get_forecast_client


@dataclass
class ForecastAgent:
    """Forecast agent with enhanced AWS Forecast capabilities."""

    orders_view_path: Path | None = None
    window_months: int = 3
    forecast_periods: int = 3
    df: pd.DataFrame = field(default=None, repr=False)
    categories: list[str] = field(default_factory=list)
    use_aws_forecast: bool = True

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
        # Store available categories for matching
        if "product_category_name_english" in self.df.columns:
            self.categories = self.df["product_category_name_english"].dropna().unique().tolist()

        # Initialize AWS Forecast client
        self._forecast_client = get_forecast_client() if self.use_aws_forecast else None

    def invoke(self, state: AgentState) -> AgentState:
        query = state.get("query", "")

        # Try to extract category and periods from query
        category = self._extract_category(query)
        periods = self._extract_periods(query) or self.forecast_periods

        try:
            # Try enhanced AWS Forecast first
            if self._forecast_client:
                forecast = self._enhanced_forecast(category, periods)
            else:
                # Fallback to original methods
                if category:
                    forecast = self._category_forecast(category)
                else:
                    forecast = self._total_forecast()
        except Exception as e:
            forecast = {
                "error": str(e),
                "method": "moving_average",
                "message": "Could not generate forecast. Using fallback."
            }

        state["forecast_result"] = forecast
        agents_used = state.get("agents_used", [])
        agents_used.append("forecast")
        state["agents_used"] = agents_used
        return state

    def _extract_periods(self, query: str) -> int | None:
        """Extract forecast periods from query."""
        import re
        # Match patterns like "next 3 months", "6 month forecast", "forecast 12 months"
        patterns = [
            r"next\s+(\d+)\s+months?",
            r"(\d+)\s+months?\s+(?:forecast|prediction)",
            r"forecast\s+(?:for\s+)?(\d+)\s+months?",
        ]
        for pattern in patterns:
            match = re.search(pattern, query.lower())
            if match:
                return min(int(match.group(1)), 12)  # Cap at 12 months
        return None

    def _enhanced_forecast(self, category: str | None, periods: int) -> dict[str, Any]:
        """Generate enhanced forecast using AWS Forecast client."""
        if category:
            # Filter data for category
            cat_df = self.df[self.df["product_category_name_english"] == category]
            if cat_df.empty:
                return {
                    "error": f"No data found for category: {category}",
                    "available_categories": self.categories[:10],
                }
            # Aggregate by month
            monthly = (
                cat_df.groupby("order_purchase_month")
                .agg(total_revenue=("price", "sum"), order_count=("order_id", "nunique"))
                .reset_index()
            )
            monthly.columns = ["date", "revenue", "orders"]
        else:
            # Total revenue forecast
            monthly = (
                self.df.groupby("order_purchase_month")
                .agg(total_revenue=("price", "sum"), order_count=("order_id", "nunique"))
                .reset_index()
            )
            monthly.columns = ["date", "revenue", "orders"]

        # Use AWS Forecast client for predictions
        result = self._forecast_client.quick_forecast(
            df=monthly,
            date_column="date",
            value_column="revenue",
            periods=periods,
            method="auto"
        )

        if not result.get("success"):
            # Fall back to original method
            return self._category_forecast(category) if category else self._total_forecast()

        # Enhance the result with additional info
        result["category"] = category or "all"
        result["periods_requested"] = periods

        # Add order forecast if we have the data
        if "orders" in monthly.columns:
            orders_result = self._forecast_client.quick_forecast(
                df=monthly,
                date_column="date",
                value_column="orders",
                periods=periods,
                method="auto"
            )
            if orders_result.get("success"):
                result["orders_forecast"] = orders_result.get("predictions", [])

        return result

    def _extract_category(self, query: str) -> str | None:
        """Extract category name from query using fuzzy matching."""
        query_lower = query.lower()

        # Common category mappings
        category_aliases = {
            "electronics": ["electronics", "electronic", "tech", "gadget"],
            "computers_accessories": ["computer", "laptop", "pc", "accessories"],
            "telephony": ["phone", "telephone", "mobile", "cell"],
            "housewares": ["housewares", "household", "home"],
            "furniture_decor": ["furniture", "decor", "decoration"],
            "health_beauty": ["health", "beauty", "cosmetic"],
            "sports_leisure": ["sports", "sport", "leisure", "fitness"],
            "toys": ["toys", "toy", "games", "game"],
            "watches_gifts": ["watches", "watch", "gifts", "gift"],
            "baby": ["baby", "infant", "kids"],
            "fashion_bags_accessories": ["fashion", "bags", "bag", "accessories"],
        }

        # Check aliases first
        for category, aliases in category_aliases.items():
            for alias in aliases:
                if alias in query_lower:
                    # Find the actual category in our data
                    for real_cat in self.categories:
                        if category in real_cat.lower() or real_cat.lower() in category:
                            return real_cat

        # Direct match against actual categories
        for cat in self.categories:
            if cat.lower() in query_lower or any(word in query_lower for word in cat.lower().split("_")):
                return cat

        return None

    def _category_forecast(self, category: str) -> dict:
        """Forecast for a specific category."""
        cat_df = self.df[self.df["product_category_name_english"] == category]

        if cat_df.empty:
            return {
                "error": f"No data found for category: {category}",
                "available_categories": self.categories[:10],
            }

        monthly = (
            cat_df.groupby("order_purchase_month")
            .agg(total_revenue=("price", "sum"), order_count=("order_id", "nunique"))
            .sort_index()
        )

        tail = monthly.tail(self.window_months)
        if tail.empty:
            return {"error": f"Not enough data for category: {category}"}

        revenue_prediction = tail["total_revenue"].mean()
        orders_prediction = tail["order_count"].mean()
        last_period = tail.index.max()
        next_period = (last_period.to_period("M") + 1).to_timestamp()

        # Build chart data for visualization
        chart_data = [
            {
                "period": row.name.strftime("%Y-%m"),
                "revenue": round(float(row["total_revenue"]), 2),
                "type": "historical",
            }
            for _, row in tail.iterrows()
        ]
        # Add forecast point
        chart_data.append({
            "period": next_period.strftime("%Y-%m"),
            "revenue": round(float(revenue_prediction), 2),
            "type": "forecast",
        })

        return {
            "category": category,
            "method": "moving_average",
            "window_months": self.window_months,
            "chart_type": "line",
            "chart_title": f"Forecast: {category}",
            "x_field": "period",
            "y_field": "revenue",
            "summary": chart_data,
            "history": [
                {
                    "period": row.name.isoformat(),
                    "revenue": round(float(row["total_revenue"]), 2),
                    "orders": int(row["order_count"]),
                }
                for _, row in tail.iterrows()
            ],
            "forecast": {
                "period_start": next_period.isoformat(),
                "predicted_revenue": round(float(revenue_prediction), 2),
                "predicted_orders": round(float(orders_prediction)),
                "currency": "BRL",
            },
        }

    def _total_forecast(self) -> dict:
        """Forecast for total revenue (all categories)."""
        monthly = (
            self.df.groupby("order_purchase_month")
            .agg(total_revenue=("price", "sum"), order_count=("order_id", "nunique"))
            .sort_index()
        )

        tail = monthly.tail(self.window_months)
        revenue_prediction = tail["total_revenue"].mean()
        orders_prediction = tail["order_count"].mean()
        last_period = tail.index.max()
        next_period = (last_period.to_period("M") + 1).to_timestamp()

        # Build chart data for visualization
        chart_data = [
            {
                "period": row.name.strftime("%Y-%m"),
                "revenue": round(float(row["total_revenue"]), 2),
                "type": "historical",
            }
            for _, row in tail.iterrows()
        ]
        # Add forecast point
        chart_data.append({
            "period": next_period.strftime("%Y-%m"),
            "revenue": round(float(revenue_prediction), 2),
            "type": "forecast",
        })

        return {
            "category": "all",
            "method": "moving_average",
            "window_months": self.window_months,
            "chart_type": "line",
            "chart_title": "Revenue Forecast (All Categories)",
            "x_field": "period",
            "y_field": "revenue",
            "summary": chart_data,
            "history": [
                {
                    "period": row.name.isoformat(),
                    "revenue": round(float(row["total_revenue"]), 2),
                    "orders": int(row["order_count"]),
                }
                for _, row in tail.iterrows()
            ],
            "forecast": {
                "period_start": next_period.isoformat(),
                "predicted_revenue": round(float(revenue_prediction), 2),
                "predicted_orders": round(float(orders_prediction)),
                "currency": "BRL",
            },
        }
