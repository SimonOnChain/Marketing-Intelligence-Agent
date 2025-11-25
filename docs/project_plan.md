# Project Plan – Marketing Intelligence Agent

This plan distills `COMPLETE_PROJECT_GUIDE.md` into actionable steps so we can execute quickly.

## Phase 0 – Repo + Tooling (Day 0)
1. Initialize Python project with `uv`, `pyproject.toml`, `.env` scaffolding.
2. Create directory tree: `src/{data,retrieval,agents,api,app}` + `tests`, `infrastructure`, `data`.
3. Add Dockerfile + compose for parity.

✅ Status: Complete in repo.

## Phase 1 – Data Foundation (Days 1-2)
1. Download Olist dataset into `data/raw/`.
2. Run `src/data/etl.py` to produce:
   - `data/processed/reviews.parquet`
   - `data/processed/orders_view.parquet`
3. Add unit tests for ETL transforms.

## Phase 2 – Retrieval Layer (Days 2-3)
1. Execute `src/retrieval/index.py`:
   - Chunk & embed reviews (MiniLM).
   - Upsert into Qdrant (`reviews` collection).
   - Persist lexical corpus JSONL for BM25-lite.
2. Implement `HybridRetriever` + `RAGChain` (done) and extend tests (Lexical corpus test already added).

## Phase 3 – Agents (Days 3-5)
1. Implement `SalesAgent` (done) – revenue aggregations via pandas.
2. Implement `SentimentAgent` (done) – leverages RAG chain.
3. Implement `ForecastAgent` (done) – moving-average baseline.
4. Wire everything via `Orchestrator` (done) using LangGraph state machine.
5. Add integration tests that spin up orchestrator with fixture data.

## Phase 4 – API Layer (Day 5)
1. Build FastAPI app (`src/api/main.py`, done).
2. Add `/query` + `/health`, include latency + cost metadata.
3. Provide Lambda handler (done) + dockerized dev env.
4. Write contract tests hitting `/query` with mocked agents.

## Phase 5 – Streamlit UI (Day 6)
1. Create `src/app/streamlit_app.py` (done) mirroring guide mockups.
2. Add charts + example question chips + metrics.
3. Connect to API via env `MARKETING_AGENT_API`.

## Phase 6 – Monitoring & Evaluation (Day 7)
1. Integrate Langfuse callback (skeleton present).
2. Add RAGAS eval harness under `src/evaluation/`.

## Phase 7 – Deployment (Days 8-9)
1. Docker image -> Fly/AWS Fargate for API.
2. Lambda fallback via `src/api/lambda_handler.py`.
3. Streamlit -> HuggingFace Spaces; set secrets.
4. Add GitHub Actions for lint/test/deploy.

## Phase 8 – Polish (Day 10)
1. README refresh with diagrams from guide.
2. Capture Langfuse dashboard screenshots.
3. Record demo video + blog post outline.

Each phase can be tracked as GitHub Projects milestones. Tests + docs gates between phases ensure we never regress.

