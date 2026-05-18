from __future__ import annotations

from collections.abc import AsyncIterator

from backend.app.repositories.session_repository import SessionRepository
from backend.app.schemas.query import QueryResponse
from backend.app.services.generation_service import GenerationService
from backend.app.services.intent_service import IntentService
from backend.app.services.retrieval_service import RetrievalService
from backend.app.services.safety_service import SafetyService
from backend.app.services.validation_service import ValidationService


class OrchestrationService:
    def __init__(
        self,
        *,
        session_repository: SessionRepository,
        intent_service: IntentService,
        retrieval_service: RetrievalService,
        generation_service: GenerationService,
        validation_service: ValidationService,
        safety_service: SafetyService,
    ):
        self.session_repository = session_repository
        self.intent_service = intent_service
        self.retrieval_service = retrieval_service
        self.generation_service = generation_service
        self.validation_service = validation_service
        self.safety_service = safety_service

    async def _analysis_with_session(self, query: str, session_id: str):
        context = await self.session_repository.get_context(session_id)
        previous_topic = context.get("topic")
        analysis = self.intent_service.analyze(query, previous_topic=previous_topic)

        await self.session_repository.save_context(
            session_id,
            {
                "topic": analysis.topic,
                "intent": analysis.intent.value,
                "urgency": analysis.urgency.value,
            },
        )
        return analysis

    async def process_query(self, *, query: str, session_id: str, trace_id: str) -> QueryResponse:
        cleaned = self.safety_service.sanitize_query(query)
        analysis = await self._analysis_with_session(cleaned, session_id)

        retrieval = await self.retrieval_service.retrieve(query=cleaned, analysis=analysis)

        answer, citations = await self.generation_service.generate(
            query=cleaned,
            retrieval=retrieval,
            analysis=analysis,
        )

        validation = self.validation_service.validate(
            answer=answer,
            analysis=analysis,
            retrieval=retrieval,
        )

        final_answer = answer
        if not validation.valid:
            final_answer = (
                "I do not have enough grounded information yet. Could you share more specifics?"
            )

        return QueryResponse(
            answer=final_answer,
            confidence=validation.confidence,
            intent=analysis.intent,
            urgency=analysis.urgency,
            complexity=analysis.complexity,
            citations=citations,
            trace_id=trace_id,
            session_id=session_id,
        )

    async def stream_query(
        self,
        *,
        query: str,
        session_id: str,
    ) -> AsyncIterator[str]:
        cleaned = self.safety_service.sanitize_query(query)
        analysis = await self._analysis_with_session(cleaned, session_id)
        retrieval = await self.retrieval_service.retrieve(query=cleaned, analysis=analysis)

        async for token in self.generation_service.stream_answer(
            query=cleaned,
            retrieval=retrieval,
            analysis=analysis,
        ):
            yield token
