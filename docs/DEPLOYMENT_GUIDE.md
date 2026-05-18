# DEPLOYMENT_GUIDE.md

Date: 2026-05-18
Project: `dynamic-ai-customer-support`

## 1. Local development

## Prerequisites
- Python 3.11
- `uv`
- Redis (optional if `REDIS_ENABLED=false`)

## Setup
```bash
uv sync
cp .env .env.local  # optional
uv run uvicorn backend.app.main:app --reload
```

Health check:
```bash
curl http://127.0.0.1:8000/api/v1/health
```

## Offline ingestion job
```bash
uv run python -m backend.app.workers.run_ingestion
```

## Celery worker
```bash
uv run celery -A backend.app.workers.celery_app.celery_app worker --loglevel=info
```

## 2. Docker deployment

Build image:
```bash
docker build -t dynamic-ai-customer-support:latest .
```

Run stack (API + Redis + worker + Prometheus):
```bash
docker compose up --build
```

## 3. Kubernetes deployment (manifests)

Apply resources:
```bash
kubectl apply -f deploy/k8s/namespace.yaml
kubectl apply -f deploy/k8s/configmap.yaml
kubectl apply -f deploy/k8s/secret.yaml
kubectl apply -f deploy/k8s/deployment.yaml
kubectl apply -f deploy/k8s/service.yaml
kubectl apply -f deploy/k8s/hpa.yaml
```

Important:
- Replace `SECRET_KEY` and admin credentials in secrets before production rollout.
- Use sealed secrets or external secret manager in production.

## 4. Helm deployment

```bash
helm upgrade --install dynamic-ai deploy/helm/dynamic-ai-customer-support \
  --namespace dynamic-ai --create-namespace
```

Override sensitive values:
```bash
helm upgrade --install dynamic-ai deploy/helm/dynamic-ai-customer-support \
  --namespace dynamic-ai \
  --set secrets.SECRET_KEY="<strong-32+-char-secret>" \
  --set secrets.DEFAULT_ADMIN_PASSWORD="<strong-password>"
```

## 5. Observability
- Metrics endpoint: `/metrics`
- Health endpoint: `/api/v1/health`
- Prometheus sample config: `deploy/prometheus.yml`
- Structured logs emitted as JSON.
- Sentry enabled when `SENTRY_DSN` is configured.

## 6. CI/CD
GitHub Actions workflow: `.github/workflows/ci.yml`

Pipeline gates:
- dependency sync (`uv sync --frozen --extra dev`)
- dependency validation (`pip check`)
- lint (`ruff`)
- typing (`mypy`)
- tests + coverage (`pytest`)
- startup smoke (`scripts/startup_smoke.sh`)
- security scans (`bandit`, `pip-audit`)
- Docker image build

## 7. Production readiness checklist
- Set `ENVIRONMENT=prod`
- Set `SECRET_KEY` (>=32 chars, random)
- Disable default credentials and provision real admin accounts
- Configure `REDIS_URL` to managed Redis
- Configure ingress TLS and external rate limiting
- Enable Sentry DSN and metrics scraping
- Switch vector backend from `inmemory` to managed backend as traffic grows
