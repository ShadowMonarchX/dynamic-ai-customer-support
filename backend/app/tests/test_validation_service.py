from __future__ import annotations

from backend.app.schemas.contracts import (
    ComplexityLevel,
    IntentAnalysis,
    IntentLabel,
    RetrievalDocument,
    RetrievalResult,
    UrgencyLevel,
)
from backend.app.services.validation_service import ValidationService


def _analysis() -> IntentAnalysis:
    return IntentAnalysis(
        intent=IntentLabel.FAQ,
        urgency=UrgencyLevel.LOW,
        complexity=ComplexityLevel.SMALL,
        confidence=0.9,
        topic="general",
    )


def test_validation_detects_low_similarity() -> None:
    service = ValidationService()
    result = service.validate(
        answer="This is a detailed answer with many words",
        analysis=_analysis(),
        retrieval=RetrievalResult(docs=[], count=0, status="empty", top_similarity=0.0),
    )
    assert "low_similarity" in result.issues
    assert result.confidence < 0.5


def test_validation_accepts_grounded_answer() -> None:
    service = ValidationService()
    retrieval = RetrievalResult(
        docs=[
            RetrievalDocument(
                doc_id="1",
                text="You can contact support via email at support@example.com",
                source="kb",
                score=0.8,
                metadata={"topic": "contact"},
            )
        ],
        count=1,
        status="success",
        top_similarity=0.8,
    )
    result = service.validate(
        answer="You can contact support via email at support@example.com.",
        analysis=_analysis(),
        retrieval=retrieval,
    )
    assert result.valid is True
    assert result.confidence >= 0.45
