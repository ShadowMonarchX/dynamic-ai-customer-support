from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest

from backend.app.core.settings import Settings
from backend.app.observability.metrics import metrics_response
from backend.app.observability.sentry import configure_sentry
from backend.app.repositories.session_repository import SessionRepository
from backend.app.services.cache_service import CacheService


def test_metrics_response_content_type() -> None:
    response = metrics_response()
    assert response.media_type
    assert "text/plain" in response.media_type


def test_configure_sentry_disabled() -> None:
    settings = cast(Settings, SimpleNamespace(sentry_dsn=None, environment="dev"))
    assert configure_sentry(settings) is None


def test_configure_sentry_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    from backend.app.observability import sentry as sentry_module

    called: dict[str, object] = {}

    def fake_init(**kwargs: object) -> None:
        called.update(kwargs)

    monkeypatch.setattr(sentry_module.sentry_sdk, "init", fake_init)
    settings = cast(
        Settings,
        SimpleNamespace(sentry_dsn="http://public@example.com/1", environment="prod"),
    )
    middleware = configure_sentry(settings)

    assert middleware is sentry_module.SentryAsgiMiddleware
    assert called["dsn"] == settings.sentry_dsn
    assert called["environment"] == "prod"


@pytest.mark.asyncio
async def test_session_repository_handles_non_dict_payload() -> None:
    class _CacheStub:
        async def get_json(self, namespace: str, key: str):  # noqa: ANN202
            _ = (namespace, key)
            return "invalid"

        async def set_json(
            self,
            namespace: str,
            key: str,
            payload: dict[str, str],
            ttl_seconds: int,
        ) -> None:
            _ = (namespace, key, payload, ttl_seconds)

    repo = SessionRepository(cache_service=cast(CacheService, _CacheStub()), ttl_seconds=60)
    assert await repo.get_context("session-1") == {}

    await repo.save_context("session-1", {"topic": "identity"})


def test_celery_app_configuration() -> None:
    from backend.app.workers.celery_app import celery_app

    assert celery_app.main == "dynamic_ai_customer_support"
    assert celery_app.conf.task_serializer == "json"
    assert celery_app.conf.result_serializer == "json"


def test_ingestion_worker_task(monkeypatch: pytest.MonkeyPatch) -> None:
    from backend.app.workers import ingestion_worker

    class _IngestionStub:
        def __init__(self, settings: object):
            self.settings = settings

        async def rebuild_artifacts(self) -> int:
            return 7

    monkeypatch.setattr(ingestion_worker, "get_settings", lambda: object())
    monkeypatch.setattr(ingestion_worker, "IngestionService", _IngestionStub)

    assert ingestion_worker.rebuild_vector_artifacts() == 7


@pytest.mark.asyncio
async def test_run_ingestion_script(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from backend.app.workers import run_ingestion

    class _SettingsStub:
        vector_artifact_path = "artifacts.json"

    class _IngestionStub:
        def __init__(self, settings: object):
            self.settings = settings

        async def rebuild_artifacts(self) -> int:
            return 3

    monkeypatch.setattr(run_ingestion, "get_settings", lambda: _SettingsStub())
    monkeypatch.setattr(run_ingestion, "IngestionService", _IngestionStub)

    await run_ingestion._run()
    captured = capsys.readouterr()
    assert "Rebuilt 3 vector artifacts" in captured.out
