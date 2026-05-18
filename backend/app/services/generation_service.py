from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import AsyncIterator

import httpx

from backend.app.core.settings import Settings
from backend.app.observability.metrics import GENERATION_LATENCY
from backend.app.schemas.contracts import IntentAnalysis, RetrievalResult
from backend.app.schemas.query import Citation
from backend.app.services.safety_service import SafetyService


class GenerationService:
    def __init__(self, settings: Settings, safety_service: SafetyService):
        self.settings = settings
        self.safety_service = safety_service

    def _extractive_answer(
        self, query: str, retrieval: RetrievalResult, analysis: IntentAnalysis
    ) -> tuple[str, list[Citation]]:
        if analysis.intent.value == "greeting":
            return "Hi, how can I help you today?", []

        if retrieval.count == 0:
            return (
                "I do not have enough verified information yet. Could you clarify your request?",
                [],
            )

        query_terms = set(re.findall(r"[a-zA-Z0-9_]+", query.lower()))
        candidate_sentences: list[tuple[float, str, Citation]] = []
        for doc in retrieval.docs:
            for sentence in re.split(r"(?<=[.!?])\s+", doc.text):
                tokens = set(re.findall(r"[a-zA-Z0-9_]+", sentence.lower()))
                if not tokens:
                    continue
                overlap = len(tokens & query_terms) / max(1, len(query_terms))
                score = min(1.0, 0.6 * overlap + 0.4 * doc.score)
                citation = Citation(source=doc.source, score=doc.score, excerpt=sentence[:220])
                candidate_sentences.append((score, sentence.strip(), citation))

        candidate_sentences.sort(key=lambda item: item[0], reverse=True)
        selected = [item for item in candidate_sentences[:3] if item[1]]

        if not selected:
            return "I could not find grounded evidence for that request.", []

        answer = " ".join(sentence for _, sentence, _ in selected)
        answer = re.sub(r"\s+", " ", answer).strip()
        citations = [citation for _, _, citation in selected]
        return answer, citations

    async def _openai_compatible_answer(
        self,
        query: str,
        retrieval: RetrievalResult,
        analysis: IntentAnalysis,
    ) -> tuple[str, list[Citation]]:
        if not self.settings.openai_compatible_url:
            return self._extractive_answer(query, retrieval, analysis)

        context = "\n\n".join(doc.text for doc in retrieval.docs[:4])
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": "Answer strictly from the provided context. If missing, say you are unsure.",
                },
                {
                    "role": "user",
                    "content": f"Context:\n{self.safety_service.sanitize_context(context)}\n\nQuestion: {query}",
                },
            ],
            "temperature": 0.1,
        }

        headers: dict[str, str] = {"content-type": "application/json"}
        if self.settings.openai_compatible_api_key:
            headers["authorization"] = f"Bearer {self.settings.openai_compatible_api_key}"

        async with httpx.AsyncClient(timeout=self.settings.llm_timeout_seconds) as client:
            response = await client.post(
                self.settings.openai_compatible_url,
                content=json.dumps(payload),
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()

        content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        if not content:
            return self._extractive_answer(query, retrieval, analysis)

        citations = [
            Citation(source=doc.source, score=doc.score, excerpt=doc.text[:220])
            for doc in retrieval.docs[:3]
        ]
        return content, citations

    async def generate(
        self,
        *,
        query: str,
        retrieval: RetrievalResult,
        analysis: IntentAnalysis,
    ) -> tuple[str, list[Citation]]:
        start = time.perf_counter()

        async def _run_generation() -> tuple[str, list[Citation]]:
            if self.settings.llm_backend == "openai_compatible":
                return await self._openai_compatible_answer(query, retrieval, analysis)
            return self._extractive_answer(query, retrieval, analysis)

        answer, citations = await asyncio.wait_for(
            _run_generation(), timeout=self.settings.llm_timeout_seconds
        )
        answer = self.safety_service.moderate_output(answer)
        GENERATION_LATENCY.observe(time.perf_counter() - start)
        return answer, citations

    async def stream_answer(
        self,
        *,
        query: str,
        retrieval: RetrievalResult,
        analysis: IntentAnalysis,
    ) -> AsyncIterator[str]:
        answer, citations = await self.generate(query=query, retrieval=retrieval, analysis=analysis)
        citation_suffix = ""
        if citations:
            citation_suffix = "\n\nSources: " + "; ".join(
                f"{c.source} ({c.score:.2f})" for c in citations[:3]
            )
        full = f"{answer}{citation_suffix}".strip()
        for token in full.split(" "):
            yield token + " "
            await asyncio.sleep(0)
