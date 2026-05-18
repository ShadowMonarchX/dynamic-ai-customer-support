from __future__ import annotations

from backend.app.schemas.contracts import IntentLabel
from backend.app.services.intent_service import IntentService


def test_identity_intent_detection() -> None:
    service = IntentService()
    result = service.analyze("Who is Nayan Raval?")
    assert result.intent == IntentLabel.IDENTITY
    assert result.topic == "identity"


def test_contact_intent_detection() -> None:
    service = IntentService()
    result = service.analyze("How can I contact support by email?")
    assert result.intent == IntentLabel.CONTACT_REQUEST
    assert result.topic == "contact"


def test_transactional_intent_detection() -> None:
    service = IntentService()
    result = service.analyze("I need a refund for my order")
    assert result.intent == IntentLabel.TRANSACTIONAL
