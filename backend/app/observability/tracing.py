from __future__ import annotations

from backend.app.core.settings import Settings


def configure_tracing(settings: Settings) -> None:
    # OpenTelemetry can be auto-instrumented in deployment runtime.
    # This hook exists so startup wiring is explicit and centralized.
    _ = settings
