from __future__ import annotations

from backend.app.schemas.contracts import ComplexityLevel, IntentAnalysis, IntentLabel, UrgencyLevel


class IntentService:
    _urgent_keywords = {"urgent", "asap", "immediately", "right away", "now", "today"}
    _contact_keywords = {"contact", "email", "phone", "call", "reach"}
    _transaction_keywords = {"refund", "cancel", "payment", "billing", "invoice", "order", "track"}
    _identity_keywords = {"who is", "profile", "bio", "biography"}
    _service_keywords = {"service", "offer", "features", "capability", "expertise"}

    def analyze(self, query: str, previous_topic: str | None = None) -> IntentAnalysis:
        lowered = query.lower().strip()
        words = lowered.split()

        intent = IntentLabel.UNKNOWN
        topic = previous_topic or "general"

        if lowered in {"hi", "hello", "hey", "hii", "heyy", "yo"}:
            intent = IntentLabel.GREETING
            topic = "general"
        elif any(keyword in lowered for keyword in self._identity_keywords):
            intent = IntentLabel.IDENTITY
            topic = "identity"
        elif any(keyword in lowered for keyword in self._contact_keywords):
            intent = IntentLabel.CONTACT_REQUEST
            topic = "contact"
        elif any(keyword in lowered for keyword in self._transaction_keywords):
            intent = IntentLabel.TRANSACTIONAL
            topic = "billing" if "billing" in lowered or "invoice" in lowered else "order"
        elif any(keyword in lowered for keyword in self._service_keywords):
            intent = IntentLabel.SERVICE_QUERY
            topic = "services"
        elif lowered.endswith("?") or "how" in words or "what" in words:
            intent = IntentLabel.FAQ
            topic = topic or "general"
        else:
            intent = IntentLabel.GENERAL

        urgency = (
            UrgencyLevel.HIGH
            if any(keyword in lowered for keyword in self._urgent_keywords)
            else UrgencyLevel.LOW
        )

        if len(words) <= 5:
            complexity = ComplexityLevel.SMALL
        elif len(words) <= 20:
            complexity = ComplexityLevel.MEDIUM
        else:
            complexity = ComplexityLevel.BIG

        return IntentAnalysis(
            intent=intent,
            urgency=urgency,
            complexity=complexity,
            emotion="frustrated" if "not working" in lowered or "angry" in lowered else "neutral",
            language="en",
            confidence=0.9 if intent != IntentLabel.UNKNOWN else 0.5,
            topic=topic,
        )
