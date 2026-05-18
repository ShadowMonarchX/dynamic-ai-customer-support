# Dynamic AI Customer Support (Production Backend)

Async-first, secure, and production-ready AI customer support backend built with FastAPI.

## Highlights
- JWT auth with refresh tokens and RBAC
- Rate limiting and body-size protection
- Correlation IDs + structured JSON logging
- Redis-backed cache and session isolation with TTL
- Grounded retrieval + reranking + confidence validation
- Streaming response endpoint
- Offline ingestion worker with persisted vector artifacts
- Prometheus metrics endpoint

## Architecture

```text
backend/app/
├── api/
│   └── v1/
├── core/
├── services/
├── domain/
├── infrastructure/
├── models/
├── schemas/
├── repositories/
├── middleware/
├── workers/
├── observability/
└── tests/
```

Legacy prototype code is preserved under `experiments/legacy_backend/`.

## Environment

Configure `.env` (defaults included for local development):

- `SECRET_KEY` (required, >=32 chars)
- `DATA_PATH`
- `VECTOR_ARTIFACT_PATH`
- `REDIS_URL`
- `RATE_LIMIT`
- `VECTOR_BACKEND` (`inmemory|faiss|pgvector|pinecone|weaviate`)

## Run

```bash
uv sync
uv run uvicorn backend.app.main:app --reload
```

## Authentication

1. Get token:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/auth/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=admin123"
```

2. Query:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/query \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{"user_query":"How can I contact support?"}'
```

3. Stream query:

```bash
curl -N -X POST http://127.0.0.1:8000/api/v1/query/stream \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{"user_query":"Who is Nayan Raval?"}'
```

## Workers

Rebuild artifacts (offline ingestion job):

```bash
uv run python -m backend.app.workers.run_ingestion
```

Celery task worker:

```bash
uv run celery -A backend.app.workers.celery_app.celery_app worker --loglevel=info
```

## Metrics

- Prometheus scrape endpoint: `GET /metrics`
- Health endpoint: `GET /api/v1/health`

## Testing

```bash
uv run pytest -q
```

Coverage gate: `>= 80%`

## DevOps
- Dockerfile + docker-compose included
- GitHub Actions CI pipeline included
- Kubernetes manifests + Helm chart templates included
