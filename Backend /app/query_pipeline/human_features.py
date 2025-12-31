# app/query_pipeline/human_features.py

import threading
from typing import Dict
from langdetect import detect, DetectorFactory  # type: ignore

DetectorFactory.seed = 0


class HumanFeatureExtractor:
    URGENT_KEYWORDS = {
        "now", "urgent", "asap", "immediately", "today", "tomorrow", "right away"
    }

    FRUSTRATION_KEYWORDS = {
        "angry", "frustrated", "annoyed", "ridiculous", "worst",
        "not working", "failed"
    }

    FOLLOWUP_KEYWORDS = {
        "what about", "and then", "after that", "that", "it",
        "this", "those", "same", "continue"
    }

    # --- Shared session memory ---
    _SESSION_MEMORY: Dict[str, Dict] = {}
    _LOCK = threading.Lock()

    @classmethod
    def extract(cls, query: str, session_id: str) -> dict:
        lowered = query.lower()

        is_urgent = any(word in lowered for word in cls.URGENT_KEYWORDS)
        is_frustrated = any(word in lowered for word in cls.FRUSTRATION_KEYWORDS)
        is_followup = (
            any(word in lowered for word in cls.FOLLOWUP_KEYWORDS)
            or len(lowered.split()) <= 4
        )

        try:
            language = detect(query)
        except Exception:
            language = "unknown"

        with cls._LOCK:
            last_context = cls._SESSION_MEMORY.get(session_id, {})

            features = {
                "urgency": "high" if is_urgent else last_context.get("urgency", "low"),
                "emotion": "frustrated" if is_frustrated else last_context.get("emotion", "neutral"),
                "sentiment_score": -0.8 if is_frustrated else last_context.get("sentiment_score", 0.0),
                "language": language,
                "follow_up": is_followup,
                "previous_intent": last_context.get("intent"),
                "previous_topic": last_context.get("topic"),
            }

            cls._SESSION_MEMORY[session_id] = {
                "urgency": features["urgency"],
                "emotion": features["emotion"],
                "sentiment_score": features["sentiment_score"],
                "intent": last_context.get("intent"),
                "topic": last_context.get("topic"),
            }

        return features
