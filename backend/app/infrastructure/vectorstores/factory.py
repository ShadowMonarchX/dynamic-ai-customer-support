from __future__ import annotations

from backend.app.core.settings import Settings
from backend.app.infrastructure.vectorstores.in_memory import InMemoryVectorRepository
from backend.app.repositories.vector_repository import VectorRepository
from backend.app.services.ingestion_service import IngestionService


class FaissVectorRepository(InMemoryVectorRepository):
    def __init__(self, ingestion_service: IngestionService):
        import faiss  # noqa: F401

        super().__init__(ingestion_service)


class PgVectorRepository(InMemoryVectorRepository):
    def __init__(self, ingestion_service: IngestionService):
        import asyncpg  # noqa: F401

        super().__init__(ingestion_service)


class PineconeRepository(InMemoryVectorRepository):
    def __init__(self, ingestion_service: IngestionService):
        import pinecone  # type: ignore  # noqa: F401

        super().__init__(ingestion_service)


class WeaviateRepository(InMemoryVectorRepository):
    def __init__(self, ingestion_service: IngestionService):
        import weaviate  # type: ignore  # noqa: F401

        super().__init__(ingestion_service)


def create_vector_repository(
    settings: Settings, ingestion_service: IngestionService
) -> VectorRepository:
    backend = getattr(settings, "vector_backend", "inmemory").lower()
    if backend == "inmemory":
        return InMemoryVectorRepository(ingestion_service)
    if backend == "faiss":
        return FaissVectorRepository(ingestion_service)
    if backend == "pgvector":
        return PgVectorRepository(ingestion_service)
    if backend == "pinecone":
        return PineconeRepository(ingestion_service)
    if backend == "weaviate":
        return WeaviateRepository(ingestion_service)
    raise ValueError(f"Unsupported vector backend: {backend}")
