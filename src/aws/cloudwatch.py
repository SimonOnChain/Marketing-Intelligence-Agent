"""AWS CloudWatch integration for metrics and monitoring."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from typing import Any, Generator

import boto3
from botocore.exceptions import ClientError

from src.config.settings import get_settings


@dataclass
class MetricData:
    """Represents a CloudWatch metric data point."""
    name: str
    value: float
    unit: str = "Count"
    dimensions: dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


class CloudWatchMetrics:
    """CloudWatch metrics client for monitoring the marketing agent."""

    NAMESPACE = "MarketingIntelligenceAgent"

    # Metric names
    QUERY_COUNT = "QueryCount"
    QUERY_LATENCY = "QueryLatency"
    CACHE_HIT = "CacheHit"
    CACHE_MISS = "CacheMiss"
    AGENT_INVOCATION = "AgentInvocation"
    LLM_TOKENS = "LLMTokens"
    LLM_COST = "LLMCost"
    ERROR_COUNT = "ErrorCount"

    def __init__(self):
        """Initialize CloudWatch client."""
        settings = get_settings()

        self.client = boto3.client(
            "cloudwatch",
            region_name=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id.get_secret_value() if settings.aws_access_key_id else None,
            aws_secret_access_key=settings.aws_secret_access_key.get_secret_value() if settings.aws_secret_access_key else None,
        )
        self._enabled = bool(settings.aws_access_key_id)
        self._buffer: list[MetricData] = []
        self._buffer_size = 20  # Batch metrics for efficiency

    @property
    def enabled(self) -> bool:
        """Check if CloudWatch is available."""
        return self._enabled

    def put_metric(
        self,
        name: str,
        value: float,
        unit: str = "Count",
        dimensions: dict[str, str] | None = None,
    ) -> None:
        """Add a metric to the buffer.

        Args:
            name: Metric name
            value: Metric value
            unit: CloudWatch unit (Count, Seconds, Milliseconds, etc.)
            dimensions: Optional dimensions for the metric
        """
        if not self._enabled:
            return

        metric = MetricData(
            name=name,
            value=value,
            unit=unit,
            dimensions=dimensions or {},
        )
        self._buffer.append(metric)

        if len(self._buffer) >= self._buffer_size:
            self.flush()

    def flush(self) -> None:
        """Flush buffered metrics to CloudWatch."""
        if not self._buffer or not self._enabled:
            return

        metric_data = []
        for m in self._buffer:
            data = {
                "MetricName": m.name,
                "Value": m.value,
                "Unit": m.unit,
                "Timestamp": m.timestamp,
            }
            if m.dimensions:
                data["Dimensions"] = [
                    {"Name": k, "Value": v} for k, v in m.dimensions.items()
                ]
            metric_data.append(data)

        try:
            # CloudWatch allows max 1000 metrics per request
            for i in range(0, len(metric_data), 1000):
                batch = metric_data[i:i + 1000]
                self.client.put_metric_data(
                    Namespace=self.NAMESPACE,
                    MetricData=batch,
                )
        except ClientError as e:
            print(f"CloudWatch error: {e}")
        finally:
            self._buffer.clear()

    @contextmanager
    def measure_latency(
        self,
        metric_name: str = QUERY_LATENCY,
        dimensions: dict[str, str] | None = None,
    ) -> Generator[None, None, None]:
        """Context manager to measure and record latency.

        Usage:
            with metrics.measure_latency("QueryLatency", {"Agent": "sales"}):
                result = process_query()
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.put_metric(metric_name, elapsed_ms, "Milliseconds", dimensions)

    def record_query(
        self,
        intent: str,
        cached: bool,
        latency_ms: float,
        agents_used: list[str],
        error: bool = False,
    ) -> None:
        """Record metrics for a query execution.

        Args:
            intent: Query intent (sales, sentiment, forecast, multi)
            cached: Whether result was from cache
            latency_ms: Query latency in milliseconds
            agents_used: List of agents that were invoked
            error: Whether an error occurred
        """
        dims = {"Intent": intent}

        # Query count
        self.put_metric(self.QUERY_COUNT, 1, "Count", dims)

        # Latency
        self.put_metric(self.QUERY_LATENCY, latency_ms, "Milliseconds", dims)

        # Cache metrics
        if cached:
            self.put_metric(self.CACHE_HIT, 1, "Count")
        else:
            self.put_metric(self.CACHE_MISS, 1, "Count")

        # Agent invocations
        for agent in agents_used:
            self.put_metric(
                self.AGENT_INVOCATION,
                1,
                "Count",
                {"Agent": agent},
            )

        # Errors
        if error:
            self.put_metric(self.ERROR_COUNT, 1, "Count", dims)

    def record_llm_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
    ) -> None:
        """Record LLM token usage and cost.

        Args:
            model: Model name/ID
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cost: Estimated cost in USD
        """
        dims = {"Model": model}

        self.put_metric(self.LLM_TOKENS, input_tokens + output_tokens, "Count", dims)
        self.put_metric(self.LLM_COST, cost, "None", dims)  # USD

    def get_dashboard_url(self) -> str:
        """Get URL to CloudWatch dashboard."""
        settings = get_settings()
        region = settings.aws_region
        return f"https://{region}.console.aws.amazon.com/cloudwatch/home?region={region}#dashboards:name=MarketingAgent"

    def create_dashboard(self) -> bool:
        """Create a CloudWatch dashboard for the agent metrics."""
        if not self._enabled:
            return False

        dashboard_body = {
            "widgets": [
                {
                    "type": "metric",
                    "x": 0, "y": 0, "width": 12, "height": 6,
                    "properties": {
                        "title": "Query Volume & Latency",
                        "metrics": [
                            [self.NAMESPACE, self.QUERY_COUNT, {"stat": "Sum", "period": 300}],
                            [self.NAMESPACE, self.QUERY_LATENCY, {"stat": "Average", "period": 300, "yAxis": "right"}],
                        ],
                        "view": "timeSeries",
                        "region": get_settings().aws_region,
                    }
                },
                {
                    "type": "metric",
                    "x": 12, "y": 0, "width": 12, "height": 6,
                    "properties": {
                        "title": "Cache Performance",
                        "metrics": [
                            [self.NAMESPACE, self.CACHE_HIT, {"stat": "Sum", "period": 300}],
                            [self.NAMESPACE, self.CACHE_MISS, {"stat": "Sum", "period": 300}],
                        ],
                        "view": "timeSeries",
                        "region": get_settings().aws_region,
                    }
                },
                {
                    "type": "metric",
                    "x": 0, "y": 6, "width": 12, "height": 6,
                    "properties": {
                        "title": "Agent Invocations",
                        "metrics": [
                            [self.NAMESPACE, self.AGENT_INVOCATION, "Agent", "sales", {"stat": "Sum"}],
                            [self.NAMESPACE, self.AGENT_INVOCATION, "Agent", "sentiment", {"stat": "Sum"}],
                            [self.NAMESPACE, self.AGENT_INVOCATION, "Agent", "forecast", {"stat": "Sum"}],
                        ],
                        "view": "timeSeries",
                        "region": get_settings().aws_region,
                    }
                },
                {
                    "type": "metric",
                    "x": 12, "y": 6, "width": 12, "height": 6,
                    "properties": {
                        "title": "LLM Cost",
                        "metrics": [
                            [self.NAMESPACE, self.LLM_COST, {"stat": "Sum", "period": 3600}],
                        ],
                        "view": "timeSeries",
                        "region": get_settings().aws_region,
                    }
                },
                {
                    "type": "metric",
                    "x": 0, "y": 12, "width": 24, "height": 3,
                    "properties": {
                        "title": "Errors",
                        "metrics": [
                            [self.NAMESPACE, self.ERROR_COUNT, {"stat": "Sum", "period": 300}],
                        ],
                        "view": "singleValue",
                        "region": get_settings().aws_region,
                    }
                },
            ]
        }

        try:
            self.client.put_dashboard(
                DashboardName="MarketingAgent",
                DashboardBody=str(dashboard_body).replace("'", '"'),
            )
            return True
        except ClientError:
            return False

    def create_alarms(self) -> list[str]:
        """Create CloudWatch alarms for important metrics."""
        if not self._enabled:
            return []

        created_alarms = []
        settings = get_settings()

        alarms = [
            {
                "AlarmName": "MarketingAgent-HighLatency",
                "MetricName": self.QUERY_LATENCY,
                "Threshold": 10000,  # 10 seconds
                "ComparisonOperator": "GreaterThanThreshold",
                "EvaluationPeriods": 3,
                "Period": 300,
                "Statistic": "Average",
                "AlarmDescription": "Query latency is above 10 seconds",
            },
            {
                "AlarmName": "MarketingAgent-HighErrorRate",
                "MetricName": self.ERROR_COUNT,
                "Threshold": 10,
                "ComparisonOperator": "GreaterThanThreshold",
                "EvaluationPeriods": 2,
                "Period": 300,
                "Statistic": "Sum",
                "AlarmDescription": "Error count is too high",
            },
            {
                "AlarmName": "MarketingAgent-LowCacheHitRate",
                "MetricName": self.CACHE_MISS,
                "Threshold": 50,
                "ComparisonOperator": "GreaterThanThreshold",
                "EvaluationPeriods": 3,
                "Period": 300,
                "Statistic": "Sum",
                "AlarmDescription": "Cache miss rate is high",
            },
        ]

        for alarm in alarms:
            try:
                self.client.put_metric_alarm(
                    AlarmName=alarm["AlarmName"],
                    MetricName=alarm["MetricName"],
                    Namespace=self.NAMESPACE,
                    Threshold=alarm["Threshold"],
                    ComparisonOperator=alarm["ComparisonOperator"],
                    EvaluationPeriods=alarm["EvaluationPeriods"],
                    Period=alarm["Period"],
                    Statistic=alarm["Statistic"],
                    AlarmDescription=alarm["AlarmDescription"],
                    TreatMissingData="notBreaching",
                )
                created_alarms.append(alarm["AlarmName"])
            except ClientError:
                pass

        return created_alarms


@lru_cache(maxsize=1)
def get_cloudwatch_metrics() -> CloudWatchMetrics:
    """Get cached CloudWatch metrics client."""
    return CloudWatchMetrics()
