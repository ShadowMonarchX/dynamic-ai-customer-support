from __future__ import annotations

from typing import Protocol

from backend.app.schemas.contracts import IntentLabel, RetrievalResult


class VectorRepository(Protocol):
    async def load(self) -> None: ...

    async def search(
        self,
        *,
        query: str,
        intent: IntentLabel,
        topic: str,
        top_k: int,
    ) -> RetrievalResult: ...

    async def rebuild_from_source(self) -> int: ...
