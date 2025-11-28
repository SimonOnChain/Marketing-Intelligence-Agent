"""FastAPI surface for the Marketing Intelligence Agent."""

from __future__ import annotations

import re
import time
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from src.agents.orchestrator import Orchestrator


# ============================================
# Rate Limiting
# ============================================

class RateLimiter:
    """Simple in-memory rate limiter per IP address."""

    def __init__(self, requests_per_minute: int = 20, burst_limit: int = 5):
        self.requests_per_minute = requests_per_minute
        self.burst_limit = burst_limit
        self._requests: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, client_ip: str) -> bool:
        """Check if request is allowed for this IP."""
        now = time.time()
        window_start = now - 60  # 1 minute window

        # Clean old requests outside the window
        self._requests[client_ip] = [
            ts for ts in self._requests[client_ip] if ts > window_start
        ]

        # Check rate limit
        if len(self._requests[client_ip]) >= self.requests_per_minute:
            return False

        # Check burst (requests in last 2 seconds)
        recent = sum(1 for ts in self._requests[client_ip] if ts > now - 2)
        if recent >= self.burst_limit:
            return False

        # Allow and record
        self._requests[client_ip].append(now)
        return True

    def get_remaining(self, client_ip: str) -> int:
        """Get remaining requests for this IP."""
        now = time.time()
        window_start = now - 60
        recent = [ts for ts in self._requests.get(client_ip, []) if ts > window_start]
        return max(0, self.requests_per_minute - len(recent))


rate_limiter = RateLimiter(requests_per_minute=30, burst_limit=5)
from src.aws.cache import get_cache, InMemoryCache
from src.aws.cloudwatch import get_cloudwatch_metrics
from src.aws.redis_cache import get_redis_cache
from src.config.settings import get_settings

DATA_DIR = Path(__file__).parent.parent.parent / "data"

app = FastAPI(
    title="Marketing Intelligence Agent",
    description="Sales + Sentiment + Forecast agentic API powered by LangGraph and RAG",
    version="0.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware for frontend
# Whitelist allowed origins for security
ALLOWED_ORIGINS = [
    "http://localhost:8500",
    "http://127.0.0.1:8500",
    "http://localhost:8501",
    "http://127.0.0.1:8501",
    # Add production URLs here when deployed
    # "https://your-app.awsapprunner.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
)

# Initialize services
settings = get_settings()
query_cache = get_cache(use_dynamodb=settings.use_dynamodb_cache) if settings.cache_enabled else None
redis_cache = get_redis_cache() if settings.use_redis_cache else None
metrics = get_cloudwatch_metrics()


# ============================================
# Request/Response Models
# ============================================

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=5, max_length=1000)
    include_sources: bool = True

    @field_validator("query")
    @classmethod
    def sanitize_query(cls, v: str) -> str:
        """Sanitize and validate query input."""
        # Strip whitespace
        v = v.strip()

        # Block potential injection patterns
        dangerous_patterns = [
            r"<script",  # XSS attempt
            r"javascript:",  # XSS attempt
            r"{{.*}}",  # Template injection
            r"\$\{.*\}",  # Template injection
            r";\s*DROP",  # SQL injection
            r";\s*DELETE",  # SQL injection
            r"UNION\s+SELECT",  # SQL injection
            r"--\s*$",  # SQL comment injection
        ]
        for pattern in dangerous_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError("Invalid query: potentially malicious content detected")

        # Normalize whitespace (multiple spaces to single)
        v = re.sub(r"\s+", " ", v)

        return v


class ChartData(BaseModel):
    chart_type: str  # "bar", "line", "pie"
    chart_title: str
    x_field: str
    y_field: str
    data: list[dict]


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict] | None
    agents_used: list[str]
    execution_time: float
    cost: float
    cached: bool = False
    chart_data: ChartData | None = None


class DatasetStats(BaseModel):
    total_orders: int
    total_reviews: int
    total_products: int
    total_customers: int
    total_revenue: float
    avg_rating: float
    total_categories: int
    date_range_start: str
    date_range_end: str


class CategoryRevenue(BaseModel):
    category: str
    revenue: float


class RatingDistribution(BaseModel):
    rating: int
    count: int


class MonthlyRevenue(BaseModel):
    month: str
    revenue: float


class StateRevenue(BaseModel):
    state: str
    revenue: float
    order_count: int


class ForecastDataPoint(BaseModel):
    period: str
    value: float
    type: str  # "historical" or "forecast"


# ============================================
# Dependencies
# ============================================

@lru_cache(maxsize=1)
def get_orchestrator() -> Orchestrator:
    return Orchestrator()


@lru_cache(maxsize=1)
def get_orders_df() -> pd.DataFrame | None:
    """Load and cache orders dataframe."""
    path = DATA_DIR / "processed" / "orders_view.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None


@lru_cache(maxsize=1)
def get_reviews_df() -> pd.DataFrame | None:
    """Load and cache reviews dataframe."""
    path = DATA_DIR / "processed" / "reviews.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None


# Cached stat computations to avoid recalculating on every request
@lru_cache(maxsize=1)
def _compute_dataset_stats() -> dict:
    """Compute and cache overall dataset statistics."""
    orders_df = get_orders_df()
    reviews_df = get_reviews_df()

    stats = {
        "total_orders": 0,
        "total_reviews": 0,
        "total_products": 0,
        "total_customers": 0,
        "total_revenue": 0.0,
        "avg_rating": 0.0,
        "total_categories": 0,
        "date_range_start": "N/A",
        "date_range_end": "N/A",
    }

    if orders_df is not None:
        stats["total_orders"] = int(orders_df["order_id"].nunique()) if "order_id" in orders_df else len(orders_df)
        stats["total_products"] = int(orders_df["product_id"].nunique()) if "product_id" in orders_df else 0
        stats["total_customers"] = int(orders_df["customer_id"].nunique()) if "customer_id" in orders_df else 0
        stats["total_revenue"] = float(orders_df["price"].sum()) if "price" in orders_df else 0.0
        stats["total_categories"] = int(orders_df["product_category_name_english"].nunique()) if "product_category_name_english" in orders_df else 0

        if "order_purchase_timestamp" in orders_df:
            ts = pd.to_datetime(orders_df["order_purchase_timestamp"])
            stats["date_range_start"] = ts.min().strftime("%Y-%m-%d")
            stats["date_range_end"] = ts.max().strftime("%Y-%m-%d")

    if reviews_df is not None:
        stats["total_reviews"] = len(reviews_df)
        stats["avg_rating"] = float(reviews_df["review_score"].mean()) if "review_score" in reviews_df else 0.0

    return stats


@lru_cache(maxsize=1)
def _compute_category_revenue() -> list[tuple[str, float]]:
    """Compute and cache category revenue rankings."""
    orders_df = get_orders_df()
    if orders_df is None or "product_category_name_english" not in orders_df or "price" not in orders_df:
        return []

    return list(
        orders_df.groupby("product_category_name_english")["price"]
        .sum()
        .sort_values(ascending=False)
        .items()
    )


@lru_cache(maxsize=1)
def _compute_rating_distribution() -> list[tuple[int, int]]:
    """Compute and cache rating distribution."""
    reviews_df = get_reviews_df()
    if reviews_df is None or "review_score" not in reviews_df:
        return []

    return list(reviews_df["review_score"].value_counts().sort_index().items())


@lru_cache(maxsize=1)
def _compute_monthly_revenue() -> list[tuple[str, float]]:
    """Compute and cache monthly revenue trend."""
    orders_df = get_orders_df()
    if orders_df is None or "order_purchase_timestamp" not in orders_df or "price" not in orders_df:
        return []

    df = orders_df.copy()
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
    df["month"] = df["order_purchase_timestamp"].dt.to_period("M").astype(str)

    monthly = df.groupby("month")["price"].sum().sort_index()

    # Filter out incomplete months (less than 10% of median revenue)
    median_rev = monthly.median()
    monthly = monthly[monthly >= median_rev * 0.1]

    return list(monthly.items())


@lru_cache(maxsize=1)
def _compute_state_revenue() -> list[tuple[str, float, int]]:
    """Compute and cache revenue by Brazilian state."""
    orders_df = get_orders_df()
    if orders_df is None or "customer_state" not in orders_df or "price" not in orders_df:
        return []

    grouped = orders_df.groupby("customer_state").agg(
        revenue=("price", "sum"),
        order_count=("order_id", "nunique")
    ).reset_index()

    grouped = grouped.sort_values("revenue", ascending=False)

    return [(row["customer_state"], row["revenue"], row["order_count"])
            for _, row in grouped.iterrows()]


@lru_cache(maxsize=1)
def _compute_geo_data() -> list[dict]:
    """Compute geographic data with lat/lng for map visualization."""
    orders_df = get_orders_df()
    if orders_df is None or "customer_zip_code_prefix" not in orders_df:
        return []

    # Load geolocation data
    geo_path = DATA_DIR / "raw" / "olist_geolocation_dataset.csv"
    if not geo_path.exists():
        return []

    geo_df = pd.read_csv(geo_path)

    # Get average lat/lng per zip code prefix (there can be multiple entries)
    geo_avg = geo_df.groupby("geolocation_zip_code_prefix").agg({
        "geolocation_lat": "mean",
        "geolocation_lng": "mean",
        "geolocation_city": "first",
        "geolocation_state": "first"
    }).reset_index()

    # Aggregate orders by zip code prefix
    orders_by_zip = orders_df.groupby("customer_zip_code_prefix").agg(
        revenue=("price", "sum"),
        order_count=("order_id", "nunique")
    ).reset_index()

    # Merge with geolocation
    merged = orders_by_zip.merge(
        geo_avg,
        left_on="customer_zip_code_prefix",
        right_on="geolocation_zip_code_prefix",
        how="inner"
    )

    # Aggregate by city to reduce data points
    city_data = merged.groupby(["geolocation_city", "geolocation_state"]).agg({
        "revenue": "sum",
        "order_count": "sum",
        "geolocation_lat": "mean",
        "geolocation_lng": "mean"
    }).reset_index()

    # Sort by revenue and take top 100 cities
    city_data = city_data.sort_values("revenue", ascending=False).head(100)

    return [
        {
            "city": row["geolocation_city"],
            "state": row["geolocation_state"],
            "lat": float(row["geolocation_lat"]),
            "lng": float(row["geolocation_lng"]),
            "revenue": float(row["revenue"]),
            "orders": int(row["order_count"])
        }
        for _, row in city_data.iterrows()
    ]


@lru_cache(maxsize=1)
def _compute_forecast_demo() -> list[dict]:
    """Compute demo forecast based on historical weekly revenue using Facebook Prophet."""
    import warnings
    warnings.filterwarnings("ignore")

    orders_df = get_orders_df()
    if orders_df is None or "order_purchase_timestamp" not in orders_df or "price" not in orders_df:
        return []

    df = orders_df.copy()
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])

    # Group by week instead of month
    df["week"] = df["order_purchase_timestamp"].dt.to_period("W")

    # Count orders per week to identify incomplete weeks
    weekly_counts = df.groupby("week").size().reset_index(name="order_count")
    weekly_revenue = df.groupby("week")["price"].sum().reset_index()
    weekly_revenue.columns = ["week", "revenue"]

    weekly = weekly_revenue.merge(weekly_counts, on="week")
    weekly = weekly.sort_values("week")

    # Filter out incomplete weeks more aggressively
    median_orders = weekly["order_count"].median()
    weekly = weekly[weekly["order_count"] >= median_orders * 0.5]

    # Drop the last week if it has significantly fewer orders than previous
    if len(weekly) > 2:
        last_week_orders = weekly["order_count"].iloc[-1]
        prev_week_orders = weekly["order_count"].iloc[-2]
        if last_week_orders < prev_week_orders * 0.7:
            weekly = weekly.iloc[:-1]

    # Take last 52 weeks for better seasonality detection
    weekly = weekly.tail(52)

    if len(weekly) < 8:
        return []

    # Prepare data for Prophet (requires 'ds' and 'y' columns)
    prophet_df = pd.DataFrame({
        "ds": weekly["week"].apply(lambda x: x.start_time),
        "y": weekly["revenue"].values
    })

    result = []
    for _, row in prophet_df.iterrows():
        result.append({
            "period": row["ds"].strftime("%Y-%m-%d"),
            "value": float(row["y"]),
            "type": "historical"
        })

    # Use Prophet for forecasting with log transform for positivity
    try:
        from prophet import Prophet
        import numpy as np

        # Log transform to ensure positive forecasts
        prophet_df_log = prophet_df.copy()
        prophet_df_log["y"] = np.log1p(prophet_df_log["y"])  # log(1+y) for stability

        # Configure Prophet model
        model = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode="additive",
            changepoint_prior_scale=0.05,
            interval_width=0.8,
            growth="linear",
        )

        # Add monthly seasonality
        model.add_seasonality(name="monthly", period=30.5, fourier_order=3)

        # Fit model on log-transformed data
        model.fit(prophet_df_log)

        # Generate 8-week forecast
        future = model.make_future_dataframe(periods=8, freq="W")
        forecast = model.predict(future)

        # Extract forecast values (only future dates)
        future_forecast = forecast[forecast["ds"] > prophet_df["ds"].max()]

        for _, row in future_forecast.iterrows():
            # Inverse log transform: exp(y) - 1
            yhat = np.expm1(float(row["yhat"]))
            yhat_lower = np.expm1(float(row["yhat_lower"]))
            yhat_upper = np.expm1(float(row["yhat_upper"]))

            # Ensure reasonable bounds
            min_val = prophet_df["y"].min() * 0.5
            yhat = max(yhat, min_val)
            yhat_lower = max(yhat_lower, min_val * 0.3)
            yhat_upper = max(yhat_upper, yhat)

            result.append({
                "period": row["ds"].strftime("%Y-%m-%d"),
                "value": round(yhat, 2),
                "value_lower": round(yhat_lower, 2),
                "value_upper": round(yhat_upper, 2),
                "type": "forecast"
            })

    except Exception as e:
        # Fallback to simple Holt-Winters if Prophet fails
        import numpy as np
        revenues = prophet_df["y"].values

        alpha, beta = 0.3, 0.2
        level = revenues[0]
        trend = np.mean(np.diff(revenues[:4]))

        for i in range(1, len(revenues)):
            new_level = alpha * revenues[i] + (1 - alpha) * (level + trend)
            new_trend = beta * (new_level - level) + (1 - beta) * trend
            level, trend = new_level, new_trend

        last_week = weekly["week"].iloc[-1]
        for i in range(1, 9):
            forecast_value = level + trend * i
            forecast_week = last_week + i
            result.append({
                "period": forecast_week.start_time.strftime("%Y-%m-%d"),
                "value": round(float(forecast_value), 2),
                "type": "forecast"
            })

    return result


# ============================================
# Health & Info Endpoints
# ============================================

@app.get("/health")
def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok", "version": "0.1.0"}


@app.get("/info")
def info() -> dict:
    """API information."""
    return {
        "name": "Marketing Intelligence Agent",
        "version": "0.1.0",
        "description": "AI-powered marketing insights from 100K+ orders and reviews",
        "agents": ["sales", "sentiment", "forecast"],
        "llm": "Grok 4.1 Fast",
        "vector_db": "Qdrant",
    }


# ============================================
# Query Endpoint
# ============================================

@app.post("/query", response_model=QueryResponse)
def run_query(
    request: Request,
    payload: QueryRequest,
    orchestrator: Orchestrator = Depends(get_orchestrator)
) -> QueryResponse:
    """Run a natural language query through the agent system."""
    # Rate limit check
    client_ip = request.client.host if request.client else "unknown"
    if not rate_limiter.is_allowed(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please wait before making more requests.",
            headers={"Retry-After": "60"},
        )

    start = time.perf_counter()

    # Check Redis cache first (fastest)
    if redis_cache and redis_cache.enabled:
        cached_result = redis_cache.get_llm_response(payload.query)
        if cached_result:
            elapsed = time.perf_counter() - start
            if metrics.enabled:
                metrics.record_query(
                    intent="cached_redis",
                    cached=True,
                    latency_ms=elapsed * 1000,
                    agents_used=cached_result.get("agents_used", []),
                )
            # Reconstruct chart_data from cache if available
            cached_chart = None
            if cached_result.get("chart_data"):
                cached_chart = ChartData(**cached_result["chart_data"])
            return QueryResponse(
                answer=cached_result["answer"],
                sources=cached_result["sources"] if payload.include_sources else None,
                agents_used=cached_result.get("agents_used", []),
                execution_time=round(elapsed, 4),
                cost=0.0,
                cached=True,
                chart_data=cached_chart,
            )

    # Check DynamoDB cache (fallback)
    if query_cache:
        cached_result = query_cache.get(payload.query)
        if cached_result:
            elapsed = time.perf_counter() - start

            # Record cache hit metrics
            if metrics.enabled:
                metrics.record_query(
                    intent="cached",
                    cached=True,
                    latency_ms=elapsed * 1000,
                    agents_used=cached_result.get("agents_used", []),
                )

            # Also store in Redis for next time
            if redis_cache and redis_cache.enabled:
                redis_cache.set_llm_response(payload.query, cached_result)

            # Reconstruct chart_data from cache if available
            cached_chart = None
            if cached_result.get("chart_data"):
                cached_chart = ChartData(**cached_result["chart_data"])
            return QueryResponse(
                answer=cached_result["answer"],
                sources=cached_result["sources"] if payload.include_sources else None,
                agents_used=cached_result.get("agents_used", []),
                execution_time=round(elapsed, 2),
                cost=0.0,  # No LLM cost for cached responses
                cached=True,
                chart_data=cached_chart,
            )

    # Execute query through orchestrator
    try:
        result = orchestrator.invoke(payload.query)
    except Exception as exc:
        # Record error metrics
        if metrics.enabled:
            elapsed = time.perf_counter() - start
            metrics.record_query(
                intent="error",
                cached=False,
                latency_ms=elapsed * 1000,
                agents_used=[],
                error=True,
            )
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    elapsed = time.perf_counter() - start
    cost = estimate_cost(result)

    # Extract chart data from sales_result or forecast_result BEFORE caching
    chart_data = None
    chart_data_dict = None
    sales_result = result.get("sales_result")
    if sales_result and isinstance(sales_result, dict):
        if "chart_type" in sales_result and "summary" in sales_result:
            chart_data_dict = {
                "chart_type": sales_result.get("chart_type", "bar"),
                "chart_title": sales_result.get("chart_title", "Data"),
                "x_field": sales_result.get("x_field", ""),
                "y_field": sales_result.get("y_field", ""),
                "data": sales_result.get("summary", []),
            }
            chart_data = ChartData(**chart_data_dict)

    # Also check forecast_result
    forecast_result = result.get("forecast_result")
    if not chart_data and forecast_result and isinstance(forecast_result, dict):
        if "chart_type" in forecast_result and "summary" in forecast_result:
            chart_data_dict = {
                "chart_type": forecast_result.get("chart_type", "line"),
                "chart_title": forecast_result.get("chart_title", "Forecast"),
                "x_field": forecast_result.get("x_field", ""),
                "y_field": forecast_result.get("y_field", ""),
                "data": forecast_result.get("summary", []),
            }
            chart_data = ChartData(**chart_data_dict)

    # Cache the result in both DynamoDB and Redis (including chart_data)
    cache_data = {
        "answer": result["final_answer"],
        "sources": result.get("sources", []),
        "agents_used": result.get("agents_used", []),
        "chart_data": chart_data_dict,  # Include chart data in cache
    }

    if redis_cache and redis_cache.enabled:
        redis_cache.set_llm_response(payload.query, cache_data, ttl=settings.cache_ttl_seconds)

    if query_cache:
        query_cache.set(
            query=payload.query,
            answer=result["final_answer"],
            sources=result.get("sources", []),
            agents_used=result.get("agents_used", []),
            chart_data=chart_data_dict,  # Include chart data in cache
            ttl_seconds=settings.cache_ttl_seconds,
        )

    # Record metrics
    if metrics.enabled:
        metrics.record_query(
            intent=result.get("intent", "unknown"),
            cached=False,
            latency_ms=elapsed * 1000,
            agents_used=result.get("agents_used", []),
        )
        metrics.record_llm_usage(
            model="grok-4.1-fast",
            input_tokens=result.get("input_tokens", 0),
            output_tokens=result.get("output_tokens", 0),
            cost=cost,
        )

    response = QueryResponse(
        answer=result["final_answer"],
        sources=result["sources"] if payload.include_sources else None,
        agents_used=result.get("agents_used", []),
        execution_time=round(elapsed, 2),
        cost=cost,
        cached=False,
        chart_data=chart_data,
    )
    return response


# ============================================
# Dashboard Stats Endpoints
# ============================================

@app.get("/stats", response_model=DatasetStats)
def get_stats() -> DatasetStats:
    """Get overall dataset statistics (cached)."""
    cached = _compute_dataset_stats()
    return DatasetStats(**cached)


@app.get("/stats/categories", response_model=list[CategoryRevenue])
def get_category_revenue(limit: int = 10) -> list[CategoryRevenue]:
    """Get top categories by revenue (cached)."""
    cached = _compute_category_revenue()
    return [
        CategoryRevenue(category=cat, revenue=float(rev))
        for cat, rev in cached[:limit]
    ]


@app.get("/stats/ratings", response_model=list[RatingDistribution])
def get_rating_distribution() -> list[RatingDistribution]:
    """Get rating distribution (cached)."""
    cached = _compute_rating_distribution()
    return [
        RatingDistribution(rating=int(rating), count=int(count))
        for rating, count in cached
    ]


@app.get("/stats/monthly-revenue", response_model=list[MonthlyRevenue])
def get_monthly_revenue() -> list[MonthlyRevenue]:
    """Get monthly revenue trend (cached)."""
    cached = _compute_monthly_revenue()
    return [
        MonthlyRevenue(month=month, revenue=float(rev))
        for month, rev in cached
    ]


@app.get("/stats/by-state", response_model=list[StateRevenue])
def get_state_revenue(limit: int = 15) -> list[StateRevenue]:
    """Get revenue and orders by Brazilian state (cached)."""
    cached = _compute_state_revenue()
    return [
        StateRevenue(state=state, revenue=float(rev), order_count=int(count))
        for state, rev, count in cached[:limit]
    ]


@app.get("/stats/geo-map")
def get_geo_map_data(limit: int = 100) -> list[dict]:
    """Get geographic data with lat/lng for map visualization."""
    cached = _compute_geo_data()
    return cached[:limit]


@app.get("/stats/forecast-demo", response_model=list[ForecastDataPoint])
def get_forecast_demo() -> list[ForecastDataPoint]:
    """Get demo revenue forecast (historical + projected)."""
    cached = _compute_forecast_demo()
    return [
        ForecastDataPoint(period=item["period"], value=item["value"], type=item["type"])
        for item in cached
    ]


@lru_cache(maxsize=1)
def _compute_anomalies() -> list[dict]:
    """Detect anomalies in the data for alerts."""
    anomalies = []
    orders_df = get_orders_df()
    reviews_df = get_reviews_df()

    if orders_df is not None and "price" in orders_df.columns:
        # Check for revenue anomalies
        if "order_purchase_timestamp" in orders_df.columns:
            df = orders_df.copy()
            df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
            df["month"] = df["order_purchase_timestamp"].dt.to_period("M")

            monthly = df.groupby("month")["price"].sum().reset_index()
            monthly.columns = ["month", "revenue"]

            if len(monthly) >= 3:
                # Check last month vs average
                avg_revenue = monthly["revenue"].iloc[:-1].mean()
                std_revenue = monthly["revenue"].iloc[:-1].std()
                last_revenue = monthly["revenue"].iloc[-1]

                if std_revenue > 0:
                    z_score = (last_revenue - avg_revenue) / std_revenue

                    if z_score > 2:
                        anomalies.append({
                            "type": "positive",
                            "message": f"Revenue spike detected: {((last_revenue - avg_revenue) / avg_revenue * 100):+.1f}% above average",
                            "time": str(monthly["month"].iloc[-1]),
                        })
                    elif z_score < -2:
                        anomalies.append({
                            "type": "critical",
                            "message": f"Revenue drop detected: {((last_revenue - avg_revenue) / avg_revenue * 100):+.1f}% below average",
                            "time": str(monthly["month"].iloc[-1]),
                        })

    if reviews_df is not None and "review_score" in reviews_df.columns:
        # Check for rating anomalies
        avg_rating = reviews_df["review_score"].mean()
        recent_rating = reviews_df.tail(1000)["review_score"].mean() if len(reviews_df) > 1000 else avg_rating

        if recent_rating < avg_rating - 0.3:
            anomalies.append({
                "type": "warning",
                "message": f"Recent ratings trending down: {recent_rating:.2f} vs {avg_rating:.2f} average",
                "time": "Recent",
            })

        # Check for increase in 1-star reviews
        total_reviews = len(reviews_df)
        one_star = len(reviews_df[reviews_df["review_score"] == 1])
        one_star_pct = (one_star / total_reviews * 100) if total_reviews > 0 else 0

        if one_star_pct > 15:
            anomalies.append({
                "type": "warning",
                "message": f"High 1-star review rate: {one_star_pct:.1f}% of total reviews",
                "time": "Overall",
            })

    return anomalies


@app.get("/stats/anomalies")
def get_anomalies() -> list[dict]:
    """Get detected anomalies for dashboard alerts."""
    return _compute_anomalies()


def _compute_anomaly_detection() -> dict:
    """ML-based anomaly detection using z-scores and IQR methods."""
    import numpy as np

    orders_df = get_orders_df()
    reviews_df = get_reviews_df()

    result = {
        "revenue_anomalies": [],
        "category_anomalies": [],
        "rating_anomalies": [],
        "summary": {}
    }

    if orders_df is None:
        return result

    # 1. Revenue anomaly detection (monthly)
    if "order_purchase_timestamp" in orders_df.columns and "price" in orders_df.columns:
        df = orders_df.copy()
        df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
        df["month"] = df["order_purchase_timestamp"].dt.to_period("M").astype(str)

        monthly = df.groupby("month")["price"].sum().reset_index()
        monthly.columns = ["month", "revenue"]

        if len(monthly) >= 6:
            revenues = monthly["revenue"].values
            mean_rev = np.mean(revenues[:-1])  # Exclude last month for baseline
            std_rev = np.std(revenues[:-1])

            # Calculate z-scores
            for idx, row in monthly.iterrows():
                z_score = (row["revenue"] - mean_rev) / std_rev if std_rev > 0 else 0
                is_anomaly = bool(abs(z_score) > 2)

                result["revenue_anomalies"].append({
                    "month": row["month"],
                    "revenue": float(row["revenue"]),
                    "z_score": round(float(z_score), 2),
                    "is_anomaly": is_anomaly,
                    "type": "spike" if z_score > 2 else "drop" if z_score < -2 else "normal"
                })

            # Summary stats
            anomaly_count = sum(1 for a in result["revenue_anomalies"] if a["is_anomaly"])
            result["summary"]["revenue_anomalies_count"] = anomaly_count
            result["summary"]["revenue_mean"] = round(float(mean_rev), 2)
            result["summary"]["revenue_std"] = round(float(std_rev), 2)

    # 2. Category anomaly detection (outlier categories by performance)
    if "product_category_name_english" in orders_df.columns and "price" in orders_df.columns:
        cat_revenue = orders_df.groupby("product_category_name_english")["price"].sum()
        cat_count = orders_df.groupby("product_category_name_english").size()

        # Calculate avg revenue per order by category
        avg_per_order = (cat_revenue / cat_count).dropna()

        # IQR method for outlier detection
        q1 = avg_per_order.quantile(0.25)
        q3 = avg_per_order.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        for category, avg_rev in avg_per_order.items():
            is_high = bool(avg_rev > upper_bound)
            is_low = bool(avg_rev < lower_bound)

            if is_high or is_low:
                result["category_anomalies"].append({
                    "category": category,
                    "avg_revenue_per_order": round(float(avg_rev), 2),
                    "total_revenue": round(float(cat_revenue[category]), 2),
                    "order_count": int(cat_count[category]),
                    "type": "high_performer" if is_high else "underperformer",
                    "deviation": round(float((avg_rev - avg_per_order.median()) / avg_per_order.std()), 2) if avg_per_order.std() > 0 else 0
                })

        result["category_anomalies"].sort(key=lambda x: abs(x["deviation"]), reverse=True)
        result["summary"]["category_anomalies_count"] = len(result["category_anomalies"])

    # 3. Rating anomaly detection (categories with unusual ratings)
    if reviews_df is not None and "review_score" in reviews_df.columns:
        # Merge with orders to get category info
        if "order_id" in reviews_df.columns and "order_id" in orders_df.columns:
            merged = reviews_df.merge(
                orders_df[["order_id", "product_category_name_english"]].drop_duplicates(),
                on="order_id",
                how="inner"
            )

            if "product_category_name_english" in merged.columns:
                cat_ratings = merged.groupby("product_category_name_english")["review_score"].agg(["mean", "count"])
                cat_ratings = cat_ratings[cat_ratings["count"] >= 50]  # Min 50 reviews

                overall_mean = merged["review_score"].mean()
                overall_std = merged["review_score"].std()

                for category, row in cat_ratings.iterrows():
                    z_score = (row["mean"] - overall_mean) / overall_std if overall_std > 0 else 0

                    if abs(z_score) > 1.5:  # Significant deviation
                        result["rating_anomalies"].append({
                            "category": category,
                            "avg_rating": round(float(row["mean"]), 2),
                            "review_count": int(row["count"]),
                            "z_score": round(float(z_score), 2),
                            "type": "high_rated" if z_score > 0 else "low_rated"
                        })

                result["rating_anomalies"].sort(key=lambda x: x["z_score"])
                result["summary"]["rating_anomalies_count"] = len(result["rating_anomalies"])

    return result


@lru_cache(maxsize=1)
def _compute_price_elasticity() -> dict:
    """Estimate price elasticity by category using regression analysis."""
    import numpy as np

    orders_df = get_orders_df()
    if orders_df is None:
        return {"elasticity_by_category": [], "summary": {}}

    result = {
        "elasticity_by_category": [],
        "price_segments": [],
        "recommendations": [],
        "summary": {}
    }

    if "product_category_name_english" not in orders_df.columns or "price" not in orders_df.columns:
        return result

    # Group by category and analyze price-volume relationship
    for category in orders_df["product_category_name_english"].unique():
        cat_df = orders_df[orders_df["product_category_name_english"] == category]

        if len(cat_df) < 100:  # Need sufficient data
            continue

        prices = cat_df["price"].values

        # Create price bins and count orders in each
        try:
            price_bins = pd.qcut(prices, q=5, duplicates="drop")
            bin_counts = price_bins.value_counts().sort_index()

            if len(bin_counts) < 3:
                continue

            # Calculate pseudo-elasticity: % change in volume / % change in price
            bin_mids = [(interval.left + interval.right) / 2 for interval in bin_counts.index]
            volumes = bin_counts.values

            # Simple linear regression to estimate relationship
            log_prices = np.log(bin_mids)
            log_volumes = np.log(volumes + 1)  # +1 to avoid log(0)

            # Elasticity coefficient (slope of log-log regression)
            if len(log_prices) > 2:
                slope, intercept = np.polyfit(log_prices, log_volumes, 1)
                elasticity = round(float(slope), 2)

                # Interpret elasticity
                if elasticity < -1:
                    interpretation = "Elastic - sensitive to price changes"
                    recommendation = "Consider competitive pricing"
                elif elasticity > -0.5:
                    interpretation = "Inelastic - less sensitive to price"
                    recommendation = "Potential for premium pricing"
                else:
                    interpretation = "Unit elastic - moderate sensitivity"
                    recommendation = "Maintain current pricing strategy"

                avg_price = float(cat_df["price"].mean())
                total_revenue = float(cat_df["price"].sum())
                order_count = len(cat_df)

                result["elasticity_by_category"].append({
                    "category": category,
                    "elasticity": elasticity,
                    "interpretation": interpretation,
                    "recommendation": recommendation,
                    "avg_price": round(avg_price, 2),
                    "total_revenue": round(total_revenue, 2),
                    "order_count": order_count,
                    "price_range": {
                        "min": round(float(cat_df["price"].min()), 2),
                        "max": round(float(cat_df["price"].max()), 2)
                    }
                })
        except Exception:
            continue

    # Sort by absolute elasticity
    result["elasticity_by_category"].sort(key=lambda x: abs(x["elasticity"]), reverse=True)

    # Overall price segment analysis
    all_prices = orders_df["price"].values
    try:
        segments = pd.qcut(all_prices, q=4, labels=["Budget", "Mid-Range", "Premium", "Luxury"])
        segment_stats = orders_df.groupby(segments).agg({
            "price": ["mean", "count", "sum"]
        }).reset_index()

        for idx, row in segment_stats.iterrows():
            result["price_segments"].append({
                "segment": str(row.iloc[0]),
                "avg_price": round(float(row[("price", "mean")]), 2),
                "order_count": int(row[("price", "count")]),
                "revenue": round(float(row[("price", "sum")]), 2)
            })
    except Exception:
        pass

    # Summary
    elastic_count = sum(1 for e in result["elasticity_by_category"] if e["elasticity"] < -1)
    inelastic_count = sum(1 for e in result["elasticity_by_category"] if e["elasticity"] > -0.5)

    result["summary"] = {
        "categories_analyzed": len(result["elasticity_by_category"]),
        "elastic_categories": elastic_count,
        "inelastic_categories": inelastic_count,
        "avg_elasticity": round(np.mean([e["elasticity"] for e in result["elasticity_by_category"]]), 2) if result["elasticity_by_category"] else 0
    }

    return result


@lru_cache(maxsize=1)
def _compute_churn_indicators() -> dict:
    """Compute churn risk indicators based on customer behavior patterns."""
    import numpy as np

    orders_df = get_orders_df()
    reviews_df = get_reviews_df()

    if orders_df is None:
        return {"risk_segments": [], "indicators": [], "summary": {}}

    result = {
        "risk_segments": [],
        "indicators": [],
        "at_risk_revenue": 0,
        "summary": {}
    }

    if "customer_id" not in orders_df.columns or "order_purchase_timestamp" not in orders_df.columns:
        return result

    df = orders_df.copy()
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])

    # Calculate customer-level metrics (RFM-style)
    max_date = df["order_purchase_timestamp"].max()

    customer_metrics = df.groupby("customer_id").agg({
        "order_purchase_timestamp": ["max", "count"],
        "price": ["sum", "mean"]
    }).reset_index()

    customer_metrics.columns = ["customer_id", "last_order", "order_count", "total_spent", "avg_order"]
    customer_metrics["days_since_last"] = (max_date - customer_metrics["last_order"]).dt.days

    # Define churn risk based on recency and frequency
    # High risk: No order in 90+ days AND only 1 order
    # Medium risk: No order in 60+ days OR declining frequency
    # Low risk: Recent orders AND multiple purchases

    def calculate_risk(row):
        days = row["days_since_last"]
        orders = row["order_count"]

        if days > 90 and orders == 1:
            return "high"
        elif days > 60 or (days > 30 and orders == 1):
            return "medium"
        else:
            return "low"

    customer_metrics["risk_level"] = customer_metrics.apply(calculate_risk, axis=1)

    # Segment analysis
    for risk in ["high", "medium", "low"]:
        segment = customer_metrics[customer_metrics["risk_level"] == risk]
        if len(segment) > 0:
            result["risk_segments"].append({
                "risk_level": risk,
                "customer_count": len(segment),
                "avg_lifetime_value": round(float(segment["total_spent"].mean()), 2),
                "total_revenue_at_risk": round(float(segment["total_spent"].sum()), 2),
                "avg_days_inactive": round(float(segment["days_since_last"].mean()), 0),
                "avg_orders": round(float(segment["order_count"].mean()), 1)
            })

    # Calculate at-risk revenue
    high_risk = customer_metrics[customer_metrics["risk_level"] == "high"]
    medium_risk = customer_metrics[customer_metrics["risk_level"] == "medium"]
    result["at_risk_revenue"] = round(
        float(high_risk["total_spent"].sum() * 0.8 + medium_risk["total_spent"].sum() * 0.3), 2
    )

    # Add rating-based churn indicators if available
    if reviews_df is not None and "review_score" in reviews_df.columns:
        # Customers with low ratings are more likely to churn
        low_rating_customers = reviews_df[reviews_df["review_score"] <= 2]["order_id"].unique()

        if "order_id" in orders_df.columns:
            low_rating_orders = orders_df[orders_df["order_id"].isin(low_rating_customers)]
            low_rating_revenue = low_rating_orders["price"].sum() if len(low_rating_orders) > 0 else 0

            result["indicators"].append({
                "indicator": "Low Rating Customers",
                "description": "Customers who gave 1-2 star ratings",
                "count": int(len(low_rating_customers)),
                "revenue_impact": round(float(low_rating_revenue), 2),
                "risk_factor": "high"
            })

    # One-time buyer indicator
    one_time_buyers = customer_metrics[customer_metrics["order_count"] == 1]
    result["indicators"].append({
        "indicator": "One-Time Buyers",
        "description": "Customers with only 1 order",
        "count": len(one_time_buyers),
        "percentage": round(len(one_time_buyers) / len(customer_metrics) * 100, 1),
        "avg_order_value": round(float(one_time_buyers["total_spent"].mean()), 2),
        "risk_factor": "medium"
    })

    # Summary
    result["summary"] = {
        "total_customers": len(customer_metrics),
        "high_risk_count": len(customer_metrics[customer_metrics["risk_level"] == "high"]),
        "high_risk_pct": round(len(customer_metrics[customer_metrics["risk_level"] == "high"]) / len(customer_metrics) * 100, 1),
        "at_risk_revenue": result["at_risk_revenue"],
        "retention_opportunity": round(result["at_risk_revenue"] * 0.3, 2)  # Estimate 30% recoverable
    }

    return result


@lru_cache(maxsize=1)
def _compute_product_associations() -> dict:
    """Compute product recommendation patterns using association analysis."""
    import numpy as np
    from collections import defaultdict

    orders_df = get_orders_df()

    if orders_df is None:
        return {"associations": [], "category_affinity": [], "summary": {}}

    result = {
        "associations": [],
        "category_affinity": [],
        "top_bundles": [],
        "summary": {}
    }

    if "order_id" not in orders_df.columns or "product_category_name_english" not in orders_df.columns:
        return result

    # Build basket analysis at category level
    baskets = orders_df.groupby("order_id")["product_category_name_english"].apply(set).reset_index()
    baskets.columns = ["order_id", "categories"]

    # Only consider baskets with multiple categories
    multi_category_baskets = baskets[baskets["categories"].apply(len) > 1]

    if len(multi_category_baskets) < 10:
        return result

    # Calculate co-occurrence matrix
    category_counts = defaultdict(int)
    pair_counts = defaultdict(int)

    for _, row in multi_category_baskets.iterrows():
        cats = [c for c in row["categories"] if c is not None]
        for cat in cats:
            category_counts[cat] += 1

        # Count pairs
        for i in range(len(cats)):
            for j in range(i + 1, len(cats)):
                pair = tuple(sorted([cats[i], cats[j]]))
                pair_counts[pair] += 1

    total_baskets = len(baskets)

    # Calculate association metrics (support, confidence, lift)
    associations = []
    for (cat_a, cat_b), count in pair_counts.items():
        if count < 5:  # Minimum support threshold
            continue

        support = count / total_baskets
        confidence_a_to_b = count / category_counts[cat_a] if category_counts[cat_a] > 0 else 0
        confidence_b_to_a = count / category_counts[cat_b] if category_counts[cat_b] > 0 else 0

        expected = (category_counts[cat_a] / total_baskets) * (category_counts[cat_b] / total_baskets)
        lift = support / expected if expected > 0 else 0

        associations.append({
            "category_a": cat_a,
            "category_b": cat_b,
            "co_occurrence": count,
            "support": round(support * 100, 2),  # As percentage
            "confidence_a_to_b": round(confidence_a_to_b * 100, 1),
            "confidence_b_to_a": round(confidence_b_to_a * 100, 1),
            "lift": round(lift, 2)
        })

    # Sort by lift (strongest associations)
    associations.sort(key=lambda x: x["lift"], reverse=True)
    result["associations"] = associations[:20]  # Top 20

    # Category affinity heatmap data
    top_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    top_cat_names = [c[0] for c in top_categories]

    affinity_matrix = []
    for cat_a in top_cat_names:
        row = {"category": cat_a, "affinities": {}}
        for cat_b in top_cat_names:
            if cat_a == cat_b:
                row["affinities"][cat_b] = 100
            else:
                pair = tuple(sorted([cat_a, cat_b]))
                count = pair_counts.get(pair, 0)
                # Affinity score: co-occurrence / min(count_a, count_b)
                min_count = min(category_counts[cat_a], category_counts[cat_b])
                affinity = (count / min_count * 100) if min_count > 0 else 0
                row["affinities"][cat_b] = round(affinity, 1)
        affinity_matrix.append(row)

    result["category_affinity"] = affinity_matrix

    # Top bundle recommendations
    for assoc in result["associations"][:5]:
        if assoc["lift"] > 1.5:  # Only strong associations
            result["top_bundles"].append({
                "bundle": f"{assoc['category_a']} + {assoc['category_b']}",
                "recommendation": f"Customers who buy {assoc['category_a']} are {assoc['lift']:.1f}x more likely to also buy {assoc['category_b']}",
                "opportunity": "Create bundle deal" if assoc["lift"] > 2 else "Cross-sell opportunity"
            })

    # Summary
    result["summary"] = {
        "total_orders_analyzed": total_baskets,
        "multi_category_orders": len(multi_category_baskets),
        "multi_category_pct": round(len(multi_category_baskets) / total_baskets * 100, 1),
        "strong_associations": sum(1 for a in associations if a["lift"] > 2),
        "categories_analyzed": len(category_counts)
    }

    return result


@app.get("/stats/ml/anomalies")
def get_ml_anomalies() -> dict:
    """Get ML-based anomaly detection results."""
    return _compute_anomaly_detection()


@app.get("/stats/ml/price-elasticity")
def get_price_elasticity() -> dict:
    """Get price elasticity analysis by category."""
    return _compute_price_elasticity()


@app.get("/stats/ml/churn-prediction")
def get_churn_prediction() -> dict:
    """Get churn risk indicators and predictions."""
    return _compute_churn_indicators()


@app.get("/stats/ml/product-associations")
def get_product_associations() -> dict:
    """Get product recommendation patterns and associations."""
    return _compute_product_associations()


@app.get("/stats/insights")
def get_proactive_insights() -> list[dict]:
    """Get proactive insights and suggestions based on data analysis."""
    insights = []
    orders_df = get_orders_df()
    reviews_df = get_reviews_df()

    if orders_df is not None:
        # Top performing category insight
        if "product_category_name_english" in orders_df.columns and "price" in orders_df.columns:
            top_category = orders_df.groupby("product_category_name_english")["price"].sum().idxmax()
            top_revenue = orders_df.groupby("product_category_name_english")["price"].sum().max()
            insights.append({
                "type": "positive",
                "category": "sales",
                "message": f"Top category '{top_category}' generated R$ {top_revenue:,.0f} in revenue",
                "suggestion": f"Consider expanding inventory in {top_category}",
            })

        # Growth opportunity
        if "customer_state" in orders_df.columns and "price" in orders_df.columns:
            state_revenue = orders_df.groupby("customer_state")["price"].sum().sort_values()
            bottom_states = state_revenue.head(5).index.tolist()
            insights.append({
                "type": "info",
                "category": "growth",
                "message": f"States with growth potential: {', '.join(bottom_states)}",
                "suggestion": "Consider targeted marketing campaigns in these regions",
            })

    if reviews_df is not None and "review_score" in reviews_df.columns:
        # Sentiment insight
        avg_rating = reviews_df["review_score"].mean()
        five_star_pct = (len(reviews_df[reviews_df["review_score"] == 5]) / len(reviews_df) * 100) if len(reviews_df) > 0 else 0

        if avg_rating >= 4.0:
            insights.append({
                "type": "positive",
                "category": "sentiment",
                "message": f"Customer satisfaction is strong at {avg_rating:.1f} average rating",
                "suggestion": "Leverage positive reviews in marketing materials",
            })
        else:
            insights.append({
                "type": "warning",
                "category": "sentiment",
                "message": f"Average rating {avg_rating:.1f} indicates room for improvement",
                "suggestion": "Analyze negative reviews to identify common issues",
            })

    return insights


@app.get("/stats/summary")
def get_summary() -> dict:
    """Get a quick summary for display (cached)."""
    stats = _compute_dataset_stats()

    has_data = stats["total_orders"] > 0 or stats["total_reviews"] > 0
    return {
        "has_data": has_data,
        "orders": f"{stats['total_orders']:,}",
        "reviews": f"{stats['total_reviews']:,}",
        "revenue": f"R$ {stats['total_revenue']:,.0f}",
        "avg_rating": f"{stats['avg_rating']:.1f}",
    }


# ============================================
# AWS Service Endpoints
# ============================================

@app.get("/cache/stats")
def get_cache_stats() -> dict:
    """Get cache statistics."""
    stats = {
        "dynamodb": query_cache.get_stats() if query_cache else {"enabled": False},
        "redis": redis_cache.get_stats() if redis_cache else {"enabled": False},
    }
    return stats


@app.delete("/cache/clear")
def clear_cache() -> dict:
    """Clear all cached queries."""
    if not query_cache:
        return {"enabled": False, "cleared": 0}
    cleared = query_cache.clear_all()
    return {"enabled": True, "cleared": cleared}


@app.get("/metrics/dashboard-url")
def get_metrics_dashboard() -> dict:
    """Get CloudWatch dashboard URL."""
    if not metrics.enabled:
        return {"enabled": False}
    return {
        "enabled": True,
        "url": metrics.get_dashboard_url(),
    }


@app.post("/metrics/setup")
def setup_cloudwatch() -> dict:
    """Set up CloudWatch dashboard and alarms."""
    if not metrics.enabled:
        return {"enabled": False, "message": "AWS credentials not configured"}

    dashboard_created = metrics.create_dashboard()
    alarms_created = metrics.create_alarms()

    return {
        "enabled": True,
        "dashboard_created": dashboard_created,
        "alarms_created": alarms_created,
    }


@app.get("/aws/status")
def get_aws_status() -> dict:
    """Get status of all AWS services."""
    from src.aws.bedrock import get_bedrock_client
    from src.aws.cognito import get_cognito_auth
    from src.aws.s3 import get_s3_store

    bedrock = get_bedrock_client()
    cognito = get_cognito_auth()
    s3 = get_s3_store()

    return {
        "region": settings.aws_region,
        "services": {
            "dynamodb_cache": {
                "enabled": settings.use_dynamodb_cache and query_cache is not None,
                "type": "DynamoDB" if settings.use_dynamodb_cache else "InMemory",
            },
            "redis_cache": {
                "enabled": redis_cache is not None and redis_cache.enabled,
                "host": settings.redis_host,
            },
            "cloudwatch": {
                "enabled": metrics.enabled,
                "namespace": metrics.NAMESPACE if metrics.enabled else None,
            },
            "bedrock": {
                "enabled": bedrock is not None and bedrock.enabled,
                "intent_classification": settings.use_bedrock_for_intent,
                "synthesis": settings.use_bedrock_for_synthesis,
            },
            "cognito": {
                "enabled": cognito is not None and cognito.enabled,
            },
            "s3": {
                "enabled": s3 is not None and s3.enabled,
                "bucket": settings.s3_bucket,
            },
        },
    }


# ============================================
# Utilities
# ============================================

def estimate_cost(result: dict) -> float:
    """Estimate LLM cost based on token usage."""
    input_tokens = result.get("input_tokens", 0)
    output_tokens = result.get("output_tokens", 0)
    # Grok 4.1 Fast pricing
    cost = (input_tokens / 1_000_000 * 0.20) + (output_tokens / 1_000_000 * 0.50)
    return round(cost, 4)


# Flush metrics on shutdown
@app.on_event("shutdown")
def shutdown_event():
    """Flush metrics buffer on shutdown."""
    if metrics.enabled:
        metrics.flush()
