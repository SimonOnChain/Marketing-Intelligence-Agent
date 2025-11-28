"""Amazon Forecast integration for time-series predictions."""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

import boto3
import pandas as pd
from botocore.exceptions import ClientError

from src.config.settings import get_settings

DATA_DIR = Path(__file__).parent.parent.parent / "data"


class ForecastClient:
    """Client for Amazon Forecast service.

    Note: Amazon Forecast requires data to be uploaded to S3 and models to be
    trained asynchronously. This client provides both:
    1. Quick local forecasting (fallback) for immediate results
    2. AWS Forecast integration for production-grade predictions
    """

    def __init__(self):
        """Initialize Forecast clients."""
        settings = get_settings()

        self.forecast_client = boto3.client(
            "forecast",
            region_name=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id.get_secret_value() if settings.aws_access_key_id else None,
            aws_secret_access_key=settings.aws_secret_access_key.get_secret_value() if settings.aws_secret_access_key else None,
        )

        self.forecastquery_client = boto3.client(
            "forecastquery",
            region_name=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id.get_secret_value() if settings.aws_access_key_id else None,
            aws_secret_access_key=settings.aws_secret_access_key.get_secret_value() if settings.aws_secret_access_key else None,
        )

        self._enabled = bool(settings.aws_access_key_id)
        self._dataset_group_arn = None
        self._predictor_arn = None
        self._forecast_arn = None

    @property
    def enabled(self) -> bool:
        """Check if Forecast is available."""
        return self._enabled

    def quick_forecast(
        self,
        df: pd.DataFrame,
        date_column: str,
        value_column: str,
        periods: int = 3,
        method: str = "auto"
    ) -> dict[str, Any]:
        """Quick local forecast using statistical methods.

        This provides immediate results without AWS setup.

        Args:
            df: DataFrame with time series data
            date_column: Name of date column
            value_column: Name of value column to forecast
            periods: Number of periods to forecast
            method: 'moving_average', 'exponential', 'linear', or 'auto'

        Returns:
            Forecast results with predictions and confidence intervals
        """
        try:
            # Ensure datetime
            df = df.copy()
            df[date_column] = pd.to_datetime(df[date_column])
            df = df.sort_values(date_column)

            # Aggregate by month if needed
            if len(df) > 100:
                df = df.set_index(date_column).resample('M')[value_column].sum().reset_index()

            values = df[value_column].values
            dates = df[date_column].values

            # Choose best method if auto
            if method == "auto":
                method = self._select_best_method(values)

            # Generate forecast
            if method == "exponential":
                forecast_values, confidence = self._exponential_smoothing(values, periods)
            elif method == "linear":
                forecast_values, confidence = self._linear_trend(values, periods)
            else:  # moving_average
                forecast_values, confidence = self._moving_average(values, periods)

            # Generate forecast dates
            last_date = pd.Timestamp(dates[-1])
            forecast_dates = [
                (last_date + pd.DateOffset(months=i+1)).strftime("%Y-%m")
                for i in range(periods)
            ]

            # Build response
            historical = [
                {"date": pd.Timestamp(d).strftime("%Y-%m"), "value": float(v), "type": "historical"}
                for d, v in zip(dates[-6:], values[-6:])
            ]

            predictions = [
                {
                    "date": forecast_dates[i],
                    "value": float(forecast_values[i]),
                    "lower_bound": float(forecast_values[i] * (1 - confidence)),
                    "upper_bound": float(forecast_values[i] * (1 + confidence)),
                    "type": "forecast"
                }
                for i in range(periods)
            ]

            return {
                "success": True,
                "method": method,
                "historical": historical,
                "predictions": predictions,
                "summary": {
                    "next_period": predictions[0]["value"],
                    "trend": self._calculate_trend(values),
                    "confidence_level": 1 - confidence,
                },
                "chart_type": "line",
                "chart_title": "Revenue Forecast",
                "x_field": "date",
                "y_field": "value",
                "chart_data": historical + predictions,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "method": "fallback",
            }

    def _select_best_method(self, values: list) -> str:
        """Select best forecasting method based on data characteristics."""
        if len(values) < 6:
            return "moving_average"

        # Check for trend
        trend = self._calculate_trend(values)

        if abs(trend) > 0.1:  # Strong trend
            return "linear"
        elif len(values) >= 12:  # Enough data for smoothing
            return "exponential"
        else:
            return "moving_average"

    def _moving_average(self, values: list, periods: int, window: int = 3) -> tuple[list, float]:
        """Simple moving average forecast."""
        recent = values[-window:]
        avg = sum(recent) / len(recent)

        # Add slight trend if present
        if len(values) >= 6:
            trend = (values[-1] - values[-3]) / 3
        else:
            trend = 0

        forecasts = [avg + trend * (i + 1) for i in range(periods)]
        confidence = 0.15  # 15% confidence interval

        return forecasts, confidence

    def _exponential_smoothing(self, values: list, periods: int, alpha: float = 0.3) -> tuple[list, float]:
        """Simple exponential smoothing forecast."""
        # Calculate smoothed value
        smoothed = values[0]
        for v in values[1:]:
            smoothed = alpha * v + (1 - alpha) * smoothed

        # Calculate trend
        if len(values) >= 6:
            trend = (smoothed - values[-6]) / 6
        else:
            trend = 0

        forecasts = [smoothed + trend * (i + 1) for i in range(periods)]
        confidence = 0.12  # 12% confidence interval

        return forecasts, confidence

    def _linear_trend(self, values: list, periods: int) -> tuple[list, float]:
        """Linear regression forecast."""
        import numpy as np

        n = len(values)
        x = np.arange(n)
        y = np.array(values)

        # Simple linear regression
        x_mean = x.mean()
        y_mean = y.mean()

        slope = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
        intercept = y_mean - slope * x_mean

        # Forecast
        future_x = np.arange(n, n + periods)
        forecasts = (slope * future_x + intercept).tolist()

        # Calculate R-squared for confidence
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y_mean) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        confidence = 0.2 * (1 - r_squared)  # Lower confidence if poor fit

        return forecasts, confidence

    def _calculate_trend(self, values: list) -> str:
        """Calculate trend direction."""
        if len(values) < 3:
            return "stable"

        recent_avg = sum(values[-3:]) / 3
        earlier_avg = sum(values[:3]) / 3

        change = (recent_avg - earlier_avg) / earlier_avg if earlier_avg > 0 else 0

        if change > 0.1:
            return "increasing"
        elif change < -0.1:
            return "decreasing"
        else:
            return "stable"

    def forecast_revenue(self, category: str | None = None, periods: int = 3) -> dict[str, Any]:
        """Forecast revenue using local data.

        Args:
            category: Optional category filter
            periods: Number of months to forecast

        Returns:
            Forecast results
        """
        try:
            # Load orders data
            orders_path = DATA_DIR / "processed" / "orders_view.parquet"
            if not orders_path.exists():
                return {"success": False, "error": "Orders data not found"}

            df = pd.read_parquet(orders_path)

            # Filter by category if specified
            if category and "product_category_name_english" in df.columns:
                df = df[df["product_category_name_english"] == category]

            if df.empty:
                return {"success": False, "error": f"No data for category: {category}"}

            # Aggregate by month
            df["month"] = pd.to_datetime(df["order_purchase_timestamp"]).dt.to_period("M").dt.to_timestamp()
            monthly = df.groupby("month")["price"].sum().reset_index()

            # Generate forecast
            result = self.quick_forecast(
                monthly,
                date_column="month",
                value_column="price",
                periods=periods
            )

            if category:
                result["category"] = category
                result["chart_title"] = f"Forecast: {category}"
            else:
                result["category"] = "all"
                result["chart_title"] = "Revenue Forecast (All Categories)"

            return result

        except Exception as e:
            return {"success": False, "error": str(e)}

    def forecast_orders(self, periods: int = 3) -> dict[str, Any]:
        """Forecast order count."""
        try:
            orders_path = DATA_DIR / "processed" / "orders_view.parquet"
            if not orders_path.exists():
                return {"success": False, "error": "Orders data not found"}

            df = pd.read_parquet(orders_path)

            # Aggregate by month
            df["month"] = pd.to_datetime(df["order_purchase_timestamp"]).dt.to_period("M").dt.to_timestamp()
            monthly = df.groupby("month")["order_id"].nunique().reset_index()
            monthly.columns = ["month", "orders"]

            result = self.quick_forecast(
                monthly,
                date_column="month",
                value_column="orders",
                periods=periods
            )

            result["chart_title"] = "Order Count Forecast"
            return result

        except Exception as e:
            return {"success": False, "error": str(e)}

    # ========================================
    # AWS Forecast Methods (for production)
    # ========================================

    def create_dataset_group(self, name: str) -> str | None:
        """Create a dataset group in AWS Forecast."""
        if not self._enabled:
            return None

        try:
            response = self.forecast_client.create_dataset_group(
                DatasetGroupName=name,
                Domain="RETAIL"
            )
            self._dataset_group_arn = response["DatasetGroupArn"]
            return self._dataset_group_arn

        except ClientError as e:
            if "ResourceAlreadyExistsException" in str(e):
                # Get existing ARN
                response = self.forecast_client.list_dataset_groups()
                for group in response.get("DatasetGroups", []):
                    if name in group["DatasetGroupArn"]:
                        self._dataset_group_arn = group["DatasetGroupArn"]
                        return self._dataset_group_arn
            return None

    def list_predictors(self) -> list[dict]:
        """List existing predictors."""
        if not self._enabled:
            return []

        try:
            response = self.forecast_client.list_predictors()
            return response.get("Predictors", [])
        except ClientError:
            return []

    def query_forecast(self, forecast_arn: str, item_id: str) -> dict[str, Any] | None:
        """Query an existing forecast."""
        if not self._enabled:
            return None

        try:
            response = self.forecastquery_client.query_forecast(
                ForecastArn=forecast_arn,
                Filters={"item_id": item_id}
            )
            return response.get("Forecast")
        except ClientError:
            return None


@lru_cache(maxsize=1)
def get_forecast_client() -> ForecastClient | None:
    """Get cached Forecast client."""
    try:
        return ForecastClient()
    except Exception:
        return None
