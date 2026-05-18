from __future__ import annotations

import hashlib
import time

from pydantic import TypeAdapter

from backend.app.observability.metrics import RETRIEVAL_LATENCY
from backend.app.repositories.vector_repository import VectorRepository
from backend.app.schemas.contracts import IntentAnalysis, RetrievalDocument, RetrievalResult
from backend.app.services.cache_service import CacheService


class LightweightReranker:
    def score(self, query: str, doc: RetrievalDocument) -> float:
        text_lower = doc.text.lower()
        query_lower = query.lower()
        bonus = 0.0
        if query_lower in text_lower:
            bonus += 0.15
        topic_value = doc.metadata.get("topic")
        if isinstance(topic_value, str) and topic_value in query_lower:
            bonus += 0.1
        return max(0.0, min(1.0, doc.score + bonus))


class RetrievalService:
    def __init__(self, repository: VectorRepository, cache_service: CacheService):
        self.repository = repository
        self.cache_service = cache_service
        self.reranker = LightweightReranker()
        self._result_adapter = TypeAdapter(RetrievalResult)

    async def retrieve(
        self,
        *,
        query: str,
        analysis: IntentAnalysis,
        top_k: int = 5,
    ) -> RetrievalResult:
        key_raw = f"{query}|{analysis.intent.value}|{analysis.topic}|{top_k}"
        cache_key = hashlib.sha256(key_raw.encode("utf-8")).hexdigest()

        cached = await self.cache_service.get_json("retrieval", cache_key)
        if isinstance(cached, dict):
            return self._result_adapter.validate_python(cached)

        start = time.perf_counter()
        result = await self.repository.search(
            query=query,
            intent=analysis.intent,
            topic=analysis.topic,
            top_k=top_k,
        )

        reranked_docs: list[RetrievalDocument] = []
        for doc in result.docs:
            reranked_docs.append(doc.model_copy(update={"score": self.reranker.score(query, doc)}))

        reranked_docs.sort(key=lambda item: item.score, reverse=True)
        reranked_docs = reranked_docs[:top_k]
        top_similarity = reranked_docs[0].score if reranked_docs else 0.0
        final_result = RetrievalResult(
            docs=reranked_docs,
            count=len(reranked_docs),
            status="success" if reranked_docs else "empty",
            top_similarity=top_similarity,
        )
        RETRIEVAL_LATENCY.observe(time.perf_counter() - start)

        await self.cache_service.set_json(
            "retrieval",
            cache_key,
            final_result.model_dump(mode="json"),
        )
        return final_result
