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

# Context memory per session
_SESSION_CONTEXT: Dict[str, Dict[str, Any]] = {}
_LOCK = threading.Lock()


class IntentFeaturesExtractor:
    def __init__(self, max_decay: int = 5):
        self._lock = threading.Lock()
        self.max_decay = max_decay  # number of turns before context decays

    def extract(self, query: str, previous_context: Dict[str, Any] | None = None, session_id: str | None = None) -> Dict[str, Any]:
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

            # --- Session context handling ---
            if session_id:
                context = _SESSION_CONTEXT.get(session_id, {"turn_count": 0, "intent_topic": intent_topic, "question_type": question_type})

                # Follow-up chaining with topic locking
                if is_followup:
                    intent_topic = context.get("intent_topic", intent_topic)
                    question_type = context.get("question_type", question_type)
                else:
                    # Reset turn count for new topic
                    context["turn_count"] = 0

                # Increase turn count and apply context decay
                context["turn_count"] = context.get("turn_count", 0) + 1
                if context["turn_count"] > self.max_decay:
                    context["intent_topic"] = intent_topic
                    context["question_type"] = question_type
                    context["turn_count"] = 0  # reset decay

                # Save back context
                context.update({
                    "intent_topic": intent_topic,
                    "question_type": question_type,
                    "follow_up": is_followup
                })
                _SESSION_CONTEXT[session_id] = context

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
        if any(word in text for word in ["who", "what", "whose", "profile", "bio"]):
            return "identity"
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
        if any(word in text for word in ["who", "what", "whose", "bio", "profile"]):
            return "identity_query"
        return "general"

    def _default_response(self) -> Dict[str, Any]:
        return {
            "intent_topic": "general",
            "question_type": "general",
            "follow_up": False,
            "language": "unknown"
        }
