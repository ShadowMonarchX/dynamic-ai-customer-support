from __future__ import annotations

import sentry_sdk
from sentry_sdk.integrations.asgi import SentryAsgiMiddleware

from backend.app.core.settings import Settings


def configure_sentry(settings: Settings):
    if not settings.sentry_dsn:
        return None

    sentry_sdk.init(
        dsn=settings.sentry_dsn,
        environment=settings.environment,
        send_default_pii=False,
        traces_sample_rate=0.1,
    )
    return SentryAsgiMiddleware
