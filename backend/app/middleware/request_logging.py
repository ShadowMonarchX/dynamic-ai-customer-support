from __future__ import annotations

import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from backend.app.core.logging import get_logger


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.logger = get_logger("request")

    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        self.logger.info(
            "request_completed",
            extra={
                "trace_id": getattr(request.state, "trace_id", "unknown"),
                "session_id": getattr(request.state, "session_id", "unknown"),
                "path": request.url.path,
                "method": request.method,
                "elapsed_ms": round(elapsed_ms, 2),
                "status_code": response.status_code,
            },
        )
        return response
