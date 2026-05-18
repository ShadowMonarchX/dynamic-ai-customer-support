from __future__ import annotations

import re

from backend.app.schemas.contracts import IntentAnalysis, RetrievalResult, ValidationResult


class ValidationService:
    def _similarity_threshold(self, intent: str) -> float:
        mapping = {
            "identity": 0.2,
            "transactional": 0.25,
            "service_query": 0.2,
            "faq": 0.2,
            "general": 0.15,
            "unknown": 0.15,
        }
        return mapping.get(intent, 0.15)

    def _hallucination_score(self, answer: str, retrieval: RetrievalResult) -> float:
        if retrieval.count == 0:
            return 0.0
        context = " ".join(doc.text for doc in retrieval.docs).lower()
        answer_tokens = set(re.findall(r"[a-zA-Z0-9_]+", answer.lower()))
        if not answer_tokens:
            return 0.0
        context_tokens = set(re.findall(r"[a-zA-Z0-9_]+", context))
        overlap = len(answer_tokens & context_tokens) / len(answer_tokens)
        return overlap

    def validate(
        self,
        *,
        answer: str,
        analysis: IntentAnalysis,
        retrieval: RetrievalResult,
    ) -> ValidationResult:
        issues: list[str] = []
        if not answer.strip():
            issues.append("empty_answer")

        threshold = self._similarity_threshold(analysis.intent.value)
        if retrieval.top_similarity < threshold:
            issues.append("low_similarity")

        hallucination_overlap = self._hallucination_score(answer, retrieval)
        if hallucination_overlap < 0.25 and retrieval.count > 0:
            issues.append("weak_grounding")

        words = answer.split()
        if len(words) < 4:
            issues.append("too_short")
        if len(words) > 220:
            issues.append("too_long")

        relevance = max(0.0, min(1.0, retrieval.top_similarity))
        clarity = max(0.0, min(1.0, 1.0 - (0.2 if "too_long" in issues else 0.0)))
        consistency = max(0.0, min(1.0, hallucination_overlap))
        completeness = max(0.0, min(1.0, 0.8 if len(words) >= 8 else 0.5))

        penalty = 0.15 * len(issues)
        final_score = max(
            0.0, min(1.0, (relevance + clarity + consistency + completeness) / 4 - penalty)
        )
        valid = final_score >= 0.45 and "empty_answer" not in issues

        trust_message = (
            "High confidence answer."
            if final_score >= 0.8
            else "Moderate confidence answer." if final_score >= 0.5 else "Low confidence answer."
        )

        return ValidationResult(
            valid=valid,
            confidence=final_score,
            issues=issues,
            trust_message=trust_message,
            relevance=relevance,
            clarity=clarity,
            consistency=consistency,
            completeness=completeness,
            final_score=final_score,
        )
