"""Performance test for Marketing Intelligence API and UI."""

import time
import statistics
import httpx
import sys
sys.path.insert(0, ".")

# Use 127.0.0.1 for faster DNS resolution on Windows
API_URL = "http://127.0.0.1:8000"

def measure_time(func):
    """Measure execution time of a function."""
    start = time.perf_counter()
    result = func()
    elapsed = time.perf_counter() - start
    return result, elapsed


def test_health_endpoint():
    """Test health endpoint latency."""
    print("\n" + "=" * 60)
    print("1. HEALTH ENDPOINT LATENCY")
    print("=" * 60)

    times = []
    for i in range(5):
        try:
            with httpx.Client(timeout=5.0) as client:
                start = time.perf_counter()
                response = client.get(f"{API_URL}/health")
                elapsed = (time.perf_counter() - start) * 1000  # ms
                times.append(elapsed)
                print(f"   Request {i+1}: {elapsed:.1f}ms")
        except Exception as e:
            print(f"   Request {i+1}: FAILED - {e}")

    if times:
        print(f"\n   Average: {statistics.mean(times):.1f}ms")
        print(f"   Min: {min(times):.1f}ms")
        print(f"   Max: {max(times):.1f}ms")
    return times


def test_stats_endpoint():
    """Test stats endpoint latency."""
    print("\n" + "=" * 60)
    print("2. STATS ENDPOINT LATENCY")
    print("=" * 60)

    times = []
    for i in range(3):
        try:
            with httpx.Client(timeout=10.0) as client:
                start = time.perf_counter()
                response = client.get(f"{API_URL}/stats")
                elapsed = (time.perf_counter() - start) * 1000  # ms
                times.append(elapsed)
                print(f"   Request {i+1}: {elapsed:.1f}ms")
        except Exception as e:
            print(f"   Request {i+1}: FAILED - {e}")

    if times:
        print(f"\n   Average: {statistics.mean(times):.1f}ms")
    return times


def test_query_endpoint_cached():
    """Test query endpoint with likely cached response."""
    print("\n" + "=" * 60)
    print("3. QUERY ENDPOINT - CACHED RESPONSE")
    print("=" * 60)

    test_query = "What are the total sales?"

    times = []
    for i in range(3):
        try:
            with httpx.Client(timeout=60.0) as client:
                start = time.perf_counter()
                response = client.post(
                    f"{API_URL}/query",
                    json={"query": test_query, "include_sources": True}
                )
                elapsed = (time.perf_counter() - start) * 1000  # ms
                data = response.json()
                cached = data.get("cached", False)
                times.append(elapsed)
                print(f"   Request {i+1}: {elapsed:.1f}ms (cached: {cached})")
        except Exception as e:
            print(f"   Request {i+1}: FAILED - {e}")

    if times:
        print(f"\n   Average: {statistics.mean(times):.1f}ms")
    return times


def test_query_endpoint_fresh():
    """Test query endpoint with fresh query (no cache)."""
    print("\n" + "=" * 60)
    print("4. QUERY ENDPOINT - FRESH QUERY (NO CACHE)")
    print("=" * 60)

    # Use unique query to avoid cache
    test_query = f"How many orders were placed in March 2018 specifically test_{int(time.time())}"

    try:
        with httpx.Client(timeout=120.0) as client:
            start = time.perf_counter()
            response = client.post(
                f"{API_URL}/query",
                json={"query": test_query, "include_sources": True}
            )
            elapsed = (time.perf_counter() - start) * 1000  # ms
            data = response.json()
            cached = data.get("cached", False)
            exec_time = data.get("execution_time", 0)
            agents = data.get("agents_used", [])

            print(f"   Total time: {elapsed:.1f}ms")
            print(f"   API reported time: {exec_time}s")
            print(f"   Cached: {cached}")
            print(f"   Agents used: {agents}")

            return elapsed, data
    except Exception as e:
        print(f"   FAILED - {e}")
        return None, None


def test_comprehend_analysis():
    """Test Comprehend sentiment analysis latency."""
    print("\n" + "=" * 60)
    print("5. COMPREHEND SENTIMENT ANALYSIS")
    print("=" * 60)

    try:
        from src.aws.comprehend import get_comprehend_client
        client = get_comprehend_client()

        if not client or not client.enabled:
            print("   Comprehend not enabled, skipping...")
            return None

        test_texts = [
            "This product is amazing! Best purchase ever!",
            "Terrible experience. Would not recommend.",
            "It's okay, nothing special.",
        ]

        times = []
        for text in test_texts:
            start = time.perf_counter()
            result = client.analyze_sentiment(text)
            elapsed = (time.perf_counter() - start) * 1000  # ms
            times.append(elapsed)
            sentiment = result.get("sentiment", "N/A")
            source = result.get("source", "N/A")
            print(f"   '{text[:30]}...' -> {sentiment} ({elapsed:.1f}ms, source: {source})")

        print(f"\n   Average: {statistics.mean(times):.1f}ms per analysis")
        return times
    except Exception as e:
        print(f"   FAILED - {e}")
        return None


def test_data_loading():
    """Test data loading latency."""
    print("\n" + "=" * 60)
    print("6. DATA LOADING LATENCY")
    print("=" * 60)

    import pandas as pd
    from pathlib import Path

    DATA_DIR = Path("data")

    files = [
        ("orders_view.parquet", DATA_DIR / "processed" / "orders_view.parquet"),
        ("reviews.parquet", DATA_DIR / "processed" / "reviews.parquet"),
    ]

    for name, path in files:
        if path.exists():
            start = time.perf_counter()
            df = pd.read_parquet(path)
            elapsed = (time.perf_counter() - start) * 1000  # ms
            print(f"   {name}: {elapsed:.1f}ms ({len(df):,} rows)")
        else:
            print(f"   {name}: NOT FOUND")


def analyze_bottlenecks():
    """Analyze and report bottlenecks."""
    print("\n" + "=" * 60)
    print("PERFORMANCE ANALYSIS SUMMARY")
    print("=" * 60)

    print("""
   TYPICAL BOTTLENECKS:

   1. LLM API Calls (2-10 seconds)
      - Intent classification: ~500ms-2s
      - Answer synthesis: ~1-5s
      - Each agent may make LLM calls

   2. Vector Search (100-500ms)
      - Qdrant similarity search
      - Embedding generation

   3. Data Loading (50-200ms, first load)
      - Parquet files loaded once, then cached
      - Pandas operations

   4. Network Latency (10-50ms)
      - UI to API communication
      - API to external services

   OPTIMIZATION STRATEGIES:

   1. Caching (Already implemented)
      - Redis cache for fast responses
      - DynamoDB cache for persistence
      - Streamlit @st.cache_data for UI

   2. Async Operations (Potential improvement)
      - Run agents in parallel where possible
      - Async API calls

   3. Model Selection (Trade-off)
      - Use faster/smaller models for intent
      - AWS Bedrock Haiku for classification

   4. UI Optimizations
      - Lazy loading of components
      - Progressive rendering
      - Skeleton loaders during wait
    """)


def main():
    print("=" * 60)
    print("MARKETING INTELLIGENCE PERFORMANCE TEST")
    print("=" * 60)
    print(f"API URL: {API_URL}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Check if API is running
    try:
        with httpx.Client(timeout=3.0) as client:
            response = client.get(f"{API_URL}/health")
            if response.status_code != 200:
                print("\n[ERROR] API is not healthy!")
                return
    except Exception as e:
        print(f"\n[ERROR] Cannot connect to API: {e}")
        print("Make sure to run: uv run uvicorn src.api.main:app --reload --port 8000")
        return

    print("\n[OK] API is running\n")

    # Run tests
    test_health_endpoint()
    test_stats_endpoint()
    test_query_endpoint_cached()
    test_query_endpoint_fresh()
    test_comprehend_analysis()
    test_data_loading()
    analyze_bottlenecks()

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
