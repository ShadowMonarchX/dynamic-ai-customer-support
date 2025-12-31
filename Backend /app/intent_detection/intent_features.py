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

URGENT_KEYWORDS = {"now", "urgent", "asap", "immediately", "today", "tomorrow", "right away"}
FRUSTRATION_KEYWORDS = {"angry", "frustrated", "annoyed", "ridiculous", "worst", "not working", "failed"}

class IntentFeaturesExtractor:
    def __init__(self):
        self._lock = threading.Lock()

    def extract(self, query: str) -> Dict[str, Any]:
        with self._lock:
            if not query or not query.strip():
                return self._default_response()

            lowered = query.lower()
            is_urgent = any(word in lowered for word in URGENT_KEYWORDS)
            is_frustrated = any(word in lowered for word in FRUSTRATION_KEYWORDS)

            try:
                language = detect(query)
            except:
                language = "unknown"

            return {
                "intent_topic": self._detect_topic(lowered),
                "question_type": self._detect_question_type(lowered),
                "urgency": "high" if is_urgent else "low",
                "emotion": "frustrated" if is_frustrated else "neutral",
                "sentiment_score": -0.8 if is_frustrated else 0.0,
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
            "urgency": "low",
            "emotion": "neutral",
            "sentiment_score": 0.0,
            "language": "unknown"
        }
