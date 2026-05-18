from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from backend.app.api.v1.auth import router as auth_router
from backend.app.api.v1.health import router as health_router
from backend.app.api.v1.query import router as query_router
from backend.app.core.container import AppContainer
from backend.app.core.exceptions import AppError
from backend.app.core.logging import configure_logging, get_logger
from backend.app.core.settings import get_settings
from backend.app.middleware.body_limit import BodySizeLimitMiddleware
from backend.app.middleware.correlation import CorrelationIdMiddleware
from backend.app.middleware.request_logging import RequestLoggingMiddleware
from backend.app.middleware.session import SessionIdMiddleware
from backend.app.observability.metrics import MetricsMiddleware, metrics_response
from backend.app.observability.sentry import configure_sentry
from backend.app.observability.tracing import configure_tracing
from backend.app.schemas.error import ErrorDetail, ErrorEnvelope

logger = get_logger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging()
    configure_tracing(settings)

    container = AppContainer.build(settings)
    app.state.container = container
    await container.startup()

    try:
        yield
    finally:
        await container.shutdown()


def create_app() -> FastAPI:
    settings = get_settings()
    sentry_middleware = configure_sentry(settings)

    limiter = Limiter(key_func=get_remote_address, default_limits=[settings.rate_limit])

    app = FastAPI(title=settings.app_name, lifespan=lifespan)
    app.state.limiter = limiter

    app.add_middleware(CorrelationIdMiddleware)
    app.add_middleware(SessionIdMiddleware)
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.max_body_size_bytes)
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(MetricsMiddleware)
    app.add_middleware(SlowAPIMiddleware)

    if sentry_middleware is not None:
        app.add_middleware(sentry_middleware)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError):
        trace_id = getattr(request.state, "trace_id", "unknown")
        payload = ErrorEnvelope(
            error=ErrorDetail(code=exc.code, message=exc.message, trace_id=trace_id)
        )
        return JSONResponse(status_code=exc.status_code, content=payload.model_dump())

    @app.exception_handler(RateLimitExceeded)
    async def ratelimit_handler(request: Request, exc: RateLimitExceeded):
        _ = exc
        trace_id = getattr(request.state, "trace_id", "unknown")
        payload = ErrorEnvelope(
            error=ErrorDetail(
                code="RATE_LIMITED",
                message="Rate limit exceeded",
                trace_id=trace_id,
            )
        )
        return JSONResponse(status_code=429, content=payload.model_dump())

    @app.exception_handler(Exception)
    async def unhandled_error(request: Request, exc: Exception):
        trace_id = getattr(request.state, "trace_id", "unknown")
        logger.exception(
            "Unhandled exception",
            extra={
                "trace_id": trace_id,
                "path": request.url.path,
                "method": request.method,
            },
        )
        payload = ErrorEnvelope(
            error=ErrorDetail(
                code="INTERNAL_ERROR",
                message="An unexpected error occurred.",
                trace_id=trace_id,
            )
        )
        return JSONResponse(status_code=500, content=payload.model_dump())

    app.include_router(health_router, prefix=settings.api_prefix)
    app.include_router(auth_router, prefix=settings.api_prefix)
    app.include_router(query_router, prefix=settings.api_prefix)

    @app.get("/metrics")
    async def metrics_endpoint():
        return metrics_response()

    return app


app = create_app()
