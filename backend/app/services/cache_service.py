from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from redis.asyncio import Redis
from redis.exceptions import RedisError

from backend.app.core.logging import get_logger
from backend.app.core.settings import Settings
from backend.app.observability.metrics import CACHE_HIT_COUNT, CACHE_MISS_COUNT


class CacheService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = get_logger("cache")
        self._redis: Redis | None = None
        self._memory_store: dict[str, tuple[float, str]] = {}
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        if not self.settings.redis_enabled:
            self.logger.info("Redis disabled by settings; using in-memory cache")
            return
        try:
            self._redis = Redis.from_url(self.settings.redis_url, decode_responses=True)
            ping_result = self._redis.ping()
            if asyncio.iscoroutine(ping_result):
                await ping_result
            self.logger.info("Connected to Redis")
        except RedisError:
            self._redis = None
            self.logger.warning("Redis unavailable; falling back to in-memory cache")

    async def close(self) -> None:
        if self._redis is not None:
            close_result = self._redis.aclose()
            if asyncio.iscoroutine(close_result):
                await close_result
            self._redis = None

    @staticmethod
    def _cache_key(namespace: str, key: str) -> str:
        return f"{namespace}:{key}"

    async def get_json(self, namespace: str, key: str) -> Any | None:
        ckey = self._cache_key(namespace, key)
        if self._redis is not None:
            raw = await self._redis.get(ckey)
            if raw is None:
                CACHE_MISS_COUNT.labels(namespace=namespace).inc()
                return None
            CACHE_HIT_COUNT.labels(namespace=namespace).inc()
            return json.loads(raw)

        async with self._lock:
            item = self._memory_store.get(ckey)
            if item is None:
                CACHE_MISS_COUNT.labels(namespace=namespace).inc()
                return None
            expires_at, payload = item
            if time.time() > expires_at:
                self._memory_store.pop(ckey, None)
                CACHE_MISS_COUNT.labels(namespace=namespace).inc()
                return None
            CACHE_HIT_COUNT.labels(namespace=namespace).inc()
            return json.loads(payload)

    async def set_json(
        self,
        namespace: str,
        key: str,
        payload: Any,
        ttl_seconds: int | None = None,
    ) -> None:
        ckey = self._cache_key(namespace, key)
        ttl = ttl_seconds or self.settings.cache_default_ttl_seconds
        encoded = json.dumps(payload)

        if self._redis is not None:
            await self._redis.set(ckey, encoded, ex=ttl)
            return

        async with self._lock:
            self._memory_store[ckey] = (time.time() + ttl, encoded)

    async def delete(self, namespace: str, key: str) -> None:
        ckey = self._cache_key(namespace, key)
        if self._redis is not None:
            await self._redis.delete(ckey)
            return
        async with self._lock:
            self._memory_store.pop(ckey, None)
