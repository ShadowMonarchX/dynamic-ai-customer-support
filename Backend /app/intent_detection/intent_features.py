# intent_features.py
# (User Query Understanding Layer)
# Purpose
#
# Extracts intent and sentiment features from raw user queries.
#
# What Happens Here
#
# Determines:
# - User intent (billing, delivery, refund, account issue)
# - Question type (status, how-to, policy, troubleshooting)
# - Language clarity (short, vague, emotional, follow-up)
#
# Example:
# User query: "My order is delayed, why?"
# Produces:
# - Topic: Order / Delivery
# - Intent: Issue / Complaint
# - Required data: Order policy + delivery timelines
#
# This step produces intent and sentiment features, not an answer.
import threading
from typing import Dict, Any
from langdetect import detect, DetectorFactory  # type: ignore

DetectorFactory.seed = 0

FOLLOWUP_KEYWORDS = {"that", "it", "this", "those", "same", "continue", "what about", "and then"}

class IntentFeaturesExtractor:
    def __init__(self):
        self._lock = threading.Lock()

    def extract(self, query: str, previous_context: Dict[str, Any] | None = None) -> Dict[str, Any]:
        with self._lock:
            if not query or not query.strip():
                return self._default_response()

            lowered = query.lower()
            is_followup = any(k in lowered for k in FOLLOWUP_KEYWORDS) or len(lowered.split()) <= 4

            try:
                language = detect(query)
            except:
                language = "unknown"

            # Detect fresh intent
            intent_topic = self._detect_topic(lowered)
            question_type = self._detect_question_type(lowered)

            # FOLLOW-UP OVERRIDE
            if is_followup and previous_context:
                intent_topic = previous_context.get("intent_topic", intent_topic)
                question_type = previous_context.get("question_type", question_type)

            return {
                "intent_topic": intent_topic,
                "question_type": question_type,
                "follow_up": is_followup,
                "language": language
            }

    def _detect_topic(self, text: str) -> str:
        if any(word in text for word in ["refund", "return", "money back"]):
            return "billing/refund"
        if any(word in text for word in ["order", "delivery", "tracking"]):
            return "order/delivery"
        if any(word in text for word in ["account", "login", "password"]):
            return "account"
        return "general"

    def _detect_question_type(self, text: str) -> str:
        if any(word in text for word in ["how", "steps", "guide", "process"]):
            return "how-to"
        if any(word in text for word in ["status", "where", "when", "expected"]):
            return "status"
        if any(word in text for word in ["policy", "terms", "conditions"]):
            return "policy"
        if any(word in text for word in ["issue", "problem", "error", "failed"]):
            return "troubleshooting"
        return "general"

    def _default_response(self) -> Dict[str, Any]:
        return {
            "intent_topic": "general",
            "question_type": "general",
            "follow_up": False,
            "language": "unknown"
        }
