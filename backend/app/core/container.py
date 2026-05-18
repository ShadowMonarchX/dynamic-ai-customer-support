from __future__ import annotations

from dataclasses import dataclass

from backend.app.core.settings import Settings
from backend.app.infrastructure.vectorstores.factory import create_vector_repository
from backend.app.repositories.session_repository import SessionRepository
from backend.app.repositories.user_repository import UserRepository
from backend.app.services.auth_service import AuthService
from backend.app.services.cache_service import CacheService
from backend.app.services.generation_service import GenerationService
from backend.app.services.ingestion_service import IngestionService
from backend.app.services.intent_service import IntentService
from backend.app.services.orchestration_service import OrchestrationService
from backend.app.services.retrieval_service import RetrievalService
from backend.app.services.safety_service import SafetyService
from backend.app.services.validation_service import ValidationService


@dataclass(slots=True)
class AppContainer:
    settings: Settings
    cache_service: CacheService
    ingestion_service: IngestionService
    session_repository: SessionRepository
    auth_service: AuthService
    safety_service: SafetyService
    intent_service: IntentService
    retrieval_service: RetrievalService
    generation_service: GenerationService
    validation_service: ValidationService
    orchestration_service: OrchestrationService

    @classmethod
    def build(cls, settings: Settings) -> AppContainer:
        cache_service = CacheService(settings)
        ingestion_service = IngestionService(settings)
        vector_repository = create_vector_repository(settings, ingestion_service)
        user_repository = UserRepository()

        session_repository = SessionRepository(
            cache_service=cache_service,
            ttl_seconds=settings.session_ttl_seconds,
        )
        auth_service = AuthService(settings, user_repository)
        safety_service = SafetyService()
        intent_service = IntentService()
        retrieval_service = RetrievalService(vector_repository, cache_service)
        generation_service = GenerationService(settings, safety_service)
        validation_service = ValidationService()

        orchestration_service = OrchestrationService(
            session_repository=session_repository,
            intent_service=intent_service,
            retrieval_service=retrieval_service,
            generation_service=generation_service,
            validation_service=validation_service,
            safety_service=safety_service,
        )

        return cls(
            settings=settings,
            cache_service=cache_service,
            ingestion_service=ingestion_service,
            session_repository=session_repository,
            auth_service=auth_service,
            safety_service=safety_service,
            intent_service=intent_service,
            retrieval_service=retrieval_service,
            generation_service=generation_service,
            validation_service=validation_service,
            orchestration_service=orchestration_service,
        )

    async def startup(self) -> None:
        await self.cache_service.connect()
        await self.auth_service.bootstrap_default_user()
        await self.ingestion_service.ensure_artifacts()
        await self.retrieval_service.repository.load()

    async def shutdown(self) -> None:
        await self.cache_service.close()
