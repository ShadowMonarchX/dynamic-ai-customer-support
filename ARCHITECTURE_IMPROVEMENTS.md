# ARCHITECTURE_IMPROVEMENTS.md

Audit date: 2026-05-18

## Target Architecture
Adopt a production-ready layered architecture:
1. API Layer: FastAPI routers, auth middleware, request validation, response envelopes.
2. Application Layer: orchestration services (query orchestration, retrieval, validation).
3. Domain Layer: intent schema, strategy policies, confidence rules.
4. Infrastructure Layer: vector store, model clients, cache, queue, telemetry.
5. Data Layer: structured store for sessions/audit/events and managed vector index.

## Key Design Changes

### 1. Configuration and Environment Isolation
- Introduce `Settings` model using environment variables only.
- Separate configs for `dev`, `staging`, `prod`.
- Validate required settings at startup.

### 2. AI Pipeline Contract Unification
- Create shared typed models:
  - `IntentLabel`
  - `UrgencyLevel`
  - `ComplexityLevel`
  - `RetrievalResult`
  - `ValidationResult`
- Replace stringly-typed feature passing with strict objects.

### 3. Service Decomposition
Refactor into modules:
- `app/api/v1/query.py`
- `app/services/ingestion_service.py`
- `app/services/retrieval_service.py`
- `app/services/generation_service.py`
- `app/services/validation_service.py`
- `app/core/settings.py`, `app/core/logging.py`, `app/core/security.py`

### 4. Runtime Separation
- Offline ingestion job and online query serving must be separate processes.
- Persist vector artifacts and metadata snapshots.
- Use queue-backed index rebuilds.

### 5. Session and State Management
- Remove process-global mutable session memory.
- Store session context in Redis with TTL and tenant/user namespace.

### 6. Observability by Default
- Structured logs with correlation IDs.
- Metrics and tracing for each pipeline stage.
- Error tracking and alerting.

## Suggested Component Additions
- Caching: Redis
- Async jobs: Celery + RabbitMQ (or Kafka for high event throughput)
- Vector scale options: pgvector (SQL-centric), Pinecone/Weaviate (managed vector)
- Inference optimization: vLLM, ONNX Runtime, TensorRT, quantization

## Reference Request Flow (Target)
1. Authenticated request enters `/api/v1/query`.
2. Input policy checks and rate-limit enforcement.
3. Query orchestration service computes features.
4. Retrieval service fetches top-k + reranked contexts with scores.
5. Generation service produces grounded response with citations.
6. Validation service evaluates confidence and policy compliance.
7. Response envelope returned with trace ID and confidence metadata.

## Migration Plan

### Phase 0 (Stabilization)
- Dependency and config cleanup.
- Fix critical runtime and logic bugs.
- Add minimum test suite and CI.

### Phase 1 (Security + Contracts)
- Add auth/rate limiting/safe errors.
- Introduce typed shared schema for intents/features/results.

### Phase 2 (Scalability)
- Split ingestion into background job.
- Introduce Redis caching and session store.
- Add persistent vector artifact lifecycle.

### Phase 3 (Reliability)
- Full observability stack.
- Load/security testing automation.
- Canary deployment and rollback playbooks.

## Success Criteria
- Startup success in clean environment: 100%
- Query p95 latency: target < 1.5s (cached), < 3s (uncached)
- Hallucination fallback rate reduced by calibrated validation
- Zero critical vulnerabilities in CI security checks
- >=80% unit coverage and passing integration suite
