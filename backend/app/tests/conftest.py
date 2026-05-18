from __future__ import annotations

from collections.abc import AsyncIterator, Generator

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from backend.app.core.settings import get_settings
from backend.app.main import create_app


@pytest.fixture(autouse=True)
def reset_settings_cache() -> Generator[None, None, None]:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest_asyncio.fixture
async def app(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[FastAPI]:
    monkeypatch.setenv("RATE_LIMIT", "100/minute")
    monkeypatch.setenv("SECRET_KEY", "dev-secret-key-minimum-length-32!!")
    application = create_app()
    async with application.router.lifespan_context(application):
        yield application


@pytest_asyncio.fixture
async def client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as async_client:
        yield async_client


@pytest_asyncio.fixture
async def auth_token(client: AsyncClient) -> str:
    response = await client.post(
        "/api/v1/auth/token",
        data={"username": "admin", "password": "admin123"},
        headers={"content-type": "application/x-www-form-urlencoded"},
    )
    assert response.status_code == 200
    payload = response.json()
    return payload["access_token"]
