from __future__ import annotations

import time

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status_code"],
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency",
    ["method", "path"],
)
CACHE_HIT_COUNT = Counter("cache_hits_total", "Cache hits", ["namespace"])
CACHE_MISS_COUNT = Counter("cache_miss_total", "Cache misses", ["namespace"])
RETRIEVAL_LATENCY = Histogram("retrieval_latency_seconds", "Retrieval latency")
GENERATION_LATENCY = Histogram("generation_latency_seconds", "Generation latency")


class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start

        path = request.url.path
        method = request.method
        status = str(response.status_code)
        REQUEST_COUNT.labels(method=method, path=path, status_code=status).inc()
        REQUEST_LATENCY.labels(method=method, path=path).observe(elapsed)
        return response


def metrics_response() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
