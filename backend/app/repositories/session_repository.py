from __future__ import annotations

from typing import Any

from backend.app.services.cache_service import CacheService


class SessionRepository:
    def __init__(self, cache_service: CacheService, ttl_seconds: int):
        self.cache_service = cache_service
        self.ttl_seconds = ttl_seconds

    async def get_context(self, session_id: str) -> dict[str, Any]:
        payload = await self.cache_service.get_json("session", session_id)
        if not isinstance(payload, dict):
            return {}
        return payload

    async def save_context(self, session_id: str, context: dict[str, Any]) -> None:
        await self.cache_service.set_json(
            "session",
            session_id,
            context,
            ttl_seconds=self.ttl_seconds,
        )
