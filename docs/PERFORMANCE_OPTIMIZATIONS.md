# PERFORMANCE_OPTIMIZATIONS.md

Date: 2026-05-18
Project: `dynamic-ai-customer-support`

## Performance Improvements Implemented

## 1. Startup path optimization
- Removed ingestion rebuild from hot request startup path.
- Added persisted vector artifact workflow and lazy vector repository load.
- Added offline ingestion execution paths:
  - `backend/app/workers/ingestion_worker.py`
  - `backend/app/workers/run_ingestion.py`

## 2. Async-first request processing
- Endpoints and service orchestration are fully async.
- Non-blocking retrieval/generation/validation pipeline execution.
- Streaming endpoint added for incremental response delivery:
  - `POST /api/v1/query/stream`

## 3. Caching and latency reduction
- Added `CacheService` with Redis backend + in-memory fallback.
- Added retrieval result caching by deterministic query+intent+topic key.
- Added session context caching with TTL isolation.
- Added cache hit/miss metrics for tuning.

## 4. Retrieval and response quality/perf balance
- Intent/topic-aware retrieval with thresholding and reranking.
- Confidence propagation into validator and response envelope.
- Fast extractive generation mode for low-latency grounded answers.
- Optional OpenAI-compatible mode with timeout protection.

## 5. Resilience and timeout handling
- Added generation timeout guard (`asyncio.wait_for` with configurable timeout).
- Added safe fallback responses when grounding confidence is low.
- Added graceful cache backend degradation path when Redis is unavailable.

## 6. Observability for performance tuning
- Added Prometheus metrics:
  - request count
  - request latency
  - retrieval latency
  - generation latency
  - cache hit/miss
- Added `/metrics` endpoint for scrape-based dashboards.

## Load testing assets
- Locust scenario: `tests/load/locustfile.py`
- k6 scenario: `tests/load/k6.js`

## Runtime knobs available
- `RATE_LIMIT`
- `CACHE_DEFAULT_TTL_SECONDS`
- `SESSION_TTL_SECONDS`
- `LLM_TIMEOUT_SECONDS`
- `REDIS_ENABLED`
- `VECTOR_BACKEND`
- `LLM_BACKEND`

## Recommended Next Stage (Scale-up)
- Replace in-memory vector runtime with pgvector/Pinecone/Weaviate in production.
- Add distributed queue broker split (RabbitMQ/Kafka) for high-volume ingestion events.
- Add batch embedding and model warm pools for high QPS workloads.
- Add GPU inference backends (vLLM/ONNX/TensorRT) based on deployment profile.
