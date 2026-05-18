from __future__ import annotations

import pytest

from backend.app.core.settings import get_settings
from backend.app.infrastructure.vectorstores.in_memory import InMemoryVectorRepository
from backend.app.schemas.contracts import (
    ComplexityLevel,
    IntentAnalysis,
    IntentLabel,
    UrgencyLevel,
)
from backend.app.services.cache_service import CacheService
from backend.app.services.ingestion_service import IngestionService
from backend.app.services.retrieval_service import RetrievalService


@pytest.mark.asyncio
async def test_retrieval_returns_documents() -> None:
    settings = get_settings()
    cache = CacheService(settings)
    await cache.connect()

    ingestion = IngestionService(settings)
    repo = InMemoryVectorRepository(ingestion)
    service = RetrievalService(repo, cache)

    analysis = IntentAnalysis(
        intent=IntentLabel.IDENTITY,
        urgency=UrgencyLevel.LOW,
        complexity=ComplexityLevel.SMALL,
        confidence=0.9,
        topic="identity",
    )
    result = await service.retrieve(query="Who is Nayan Raval", analysis=analysis, top_k=3)
    await cache.close()

    assert result.count >= 1
    assert result.top_similarity > 0
