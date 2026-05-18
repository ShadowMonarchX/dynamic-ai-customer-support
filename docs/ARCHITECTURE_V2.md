# ARCHITECTURE_V2.md

Date: 2026-05-18
Project: `dynamic-ai-customer-support`

## Target Architecture Delivered

```text
backend/app/
├── api/v1
├── core
├── services
├── domain
├── infrastructure/vectorstores
├── models
├── schemas
├── repositories
├── middleware
├── workers
├── observability
└── tests
```

## Layer Responsibilities

## API Layer
- FastAPI routers for auth, health, query, stream.
- Auth dependency and RBAC enforcement on protected endpoints.
- Standard response/error envelopes.

## Application/Service Layer
- `orchestration_service.py`: request orchestration pipeline.
- `intent_service.py`: typed intent/urgency/complexity inference.
- `retrieval_service.py`: retriever + reranking + cache integration.
- `generation_service.py`: grounded answer generation + stream path.
- `validation_service.py`: confidence and grounding evaluation.
- `auth_service.py`: authentication and token issuance.
- `cache_service.py`: Redis/in-memory caching.
- `ingestion_service.py`: offline artifact generation.

## Domain/Contract Layer
- Centralized typed contracts in `schemas/contracts.py`.
- Unified enums and payload models across pipeline boundaries.

## Infrastructure Layer
- Vector backend factory and repository abstraction.
- Middleware for correlation/session/body-limit/logging.
- Observability instrumentation for logs/metrics/tracing/sentry.
- Worker runtime via Celery task definitions.

## Data Layer
- Persisted artifact file lifecycle for retrieval corpus.
- Session state in cache with TTL.
- User repository abstraction for auth flow.

## Request Flow (Online)
1. Request enters `/api/v1/query` with bearer token.
2. Middleware assigns `trace_id` and `session_id`.
3. Orchestration sanitizes input and resolves intent.
4. Retrieval fetches/reranks contexts and caches results.
5. Generation produces grounded answer (+citations).
6. Validation scores confidence using real retrieval similarity.
7. Response includes confidence, intent metadata, citations, trace/session IDs.

## Ingestion Flow (Offline)
1. Worker or CLI ingestion reads source data path.
2. Data is chunked and enriched with metadata.
3. Artifact snapshot persisted for runtime retrieval load.
4. Serving process loads artifact on startup/lazy search path.

## Runtime Separation
- API service handles inference/query traffic.
- Celery worker handles ingestion rebuild jobs.
- Shared broker/cache endpoint via Redis.

## Legacy Isolation
- Prior prototype modules moved out of runtime path to:
  - `experiments/legacy_backend/`

## Design Guarantees
- No process-global mutable session identifier in serving path.
- Typed interfaces between pipeline stages.
- Secure-by-default startup validation and env handling.
- Observable request lifecycle with traceability hooks.
