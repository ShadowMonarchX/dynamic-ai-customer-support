from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass

from backend.app.repositories.vector_repository import VectorRepository
from backend.app.schemas.contracts import IntentLabel, RetrievalDocument, RetrievalResult
from backend.app.services.ingestion_service import IngestionService


@dataclass(slots=True)
class _Doc:
    doc_id: str
    text: str
    source: str
    metadata: dict


class InMemoryVectorRepository(VectorRepository):
    def __init__(self, ingestion_service: IngestionService):
        self.ingestion_service = ingestion_service
        self.settings = ingestion_service.settings
        self._documents: list[_Doc] = []

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return {token for token in re.findall(r"[a-zA-Z0-9_]+", text.lower()) if len(token) > 1}

    def _score_document(
        self,
        query: str,
        query_tokens: set[str],
        doc: _Doc,
        intent: IntentLabel,
        topic: str,
    ) -> float:
        doc_tokens = self._tokenize(doc.text)
        if not query_tokens or not doc_tokens:
            return 0.0

        overlap = len(query_tokens & doc_tokens) / max(1, len(query_tokens))
        phrase_bonus = 0.2 if query.lower() in doc.text.lower() else 0.0
        topic_bonus = 0.15 if topic and doc.metadata.get("topic") == topic else 0.0

        intent_bonus = 0.0
        if intent == IntentLabel.IDENTITY and doc.metadata.get("content_type") == "identity":
            intent_bonus += 0.2
        if intent == IntentLabel.CONTACT_REQUEST and doc.metadata.get("topic") == "contact":
            intent_bonus += 0.2
        if intent == IntentLabel.TRANSACTIONAL and doc.metadata.get("category") == "billing":
            intent_bonus += 0.15

        score = overlap + phrase_bonus + topic_bonus + intent_bonus
        return max(0.0, min(1.0, score))

    def _threshold_for_intent(self, intent: IntentLabel) -> float:
        return {
            IntentLabel.GREETING: 0.0,
            IntentLabel.IDENTITY: 0.05,
            IntentLabel.CONTACT_REQUEST: 0.08,
            IntentLabel.TRANSACTIONAL: 0.1,
            IntentLabel.FAQ: 0.08,
            IntentLabel.SERVICE_QUERY: 0.08,
            IntentLabel.GENERAL: 0.05,
            IntentLabel.UNKNOWN: 0.05,
        }.get(intent, 0.05)

    async def load(self) -> None:
        await self.ingestion_service.ensure_artifacts()
        payload = json.loads(self.settings.vector_artifact_path.read_text(encoding="utf-8"))
        docs = payload.get("documents", [])
        self._documents = [
            _Doc(
                doc_id=str(item["doc_id"]),
                text=str(item["text"]),
                source=str(item.get("source", "unknown")),
                metadata=dict(item.get("metadata", {})),
            )
            for item in docs
        ]

    async def rebuild_from_source(self) -> int:
        count = await self.ingestion_service.rebuild_artifacts()
        await self.load()
        return count

    async def search(
        self,
        *,
        query: str,
        intent: IntentLabel,
        topic: str,
        top_k: int,
    ) -> RetrievalResult:
        if not self._documents:
            await self.load()

        if intent == IntentLabel.GREETING:
            return RetrievalResult(docs=[], count=0, status="skip", top_similarity=0.0)

        query_tokens = self._tokenize(query)
        scored: list[tuple[float, _Doc]] = []
        for doc in self._documents:
            score = self._score_document(query, query_tokens, doc, intent, topic)
            scored.append((score, doc))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        threshold = self._threshold_for_intent(intent)

        selected_docs: list[RetrievalDocument] = []
        for score, doc in scored:
            if score < threshold:
                continue
            selected_docs.append(
                RetrievalDocument(
                    doc_id=doc.doc_id,
                    text=doc.text,
                    source=doc.source,
                    score=score,
                    metadata=doc.metadata,
                )
            )
            if len(selected_docs) >= top_k:
                break

        if not selected_docs:
            # Safe fallback: return best scoring docs (if any)
            for score, doc in scored[:top_k]:
                if math.isclose(score, 0.0):
                    continue
                selected_docs.append(
                    RetrievalDocument(
                        doc_id=doc.doc_id,
                        text=doc.text,
                        source=doc.source,
                        score=score,
                        metadata=doc.metadata,
                    )
                )

        top_similarity = selected_docs[0].score if selected_docs else 0.0
        status = "success" if selected_docs else "empty"
        return RetrievalResult(
            docs=selected_docs,
            count=len(selected_docs),
            status=status,
            top_similarity=top_similarity,
        )
