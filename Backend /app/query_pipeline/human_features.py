
import threading
from typing import Dict
from langdetect import detect, DetectorFactory

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

    _SESSION_MEMORY: Dict[str, Dict] = {}
    _LOCK = threading.Lock()

    @classmethod
    def extract(cls, query: str, session_id: str) -> dict:
        if not query or not session_id:
            return {
                "urgency": "low",
                "emotion": "neutral",
                "sentiment_score": 0.0,
                "language": "unknown",
                "follow_up": False,
                "previous_intent": None,
                "previous_topic": None,
            }

        lowered = query.lower()

        is_urgent = any(word in lowered for word in cls.URGENT_KEYWORDS)
        is_frustrated = any(word in lowered for word in cls.FRUSTRATION_KEYWORDS)
        is_followup = any(word in lowered for word in cls.FOLLOWUP_KEYWORDS) or len(lowered.split()) <= 4

        try:
            language = detect(query)
        except Exception:
            language = "unknown"

        with cls._LOCK:
            last_context = cls._SESSION_MEMORY.get(session_id, {})

            urgency = "high" if is_urgent else last_context.get("urgency", "low")
            emotion = "frustrated" if is_frustrated else last_context.get("emotion", "neutral")
            sentiment_score = -0.8 if is_frustrated else last_context.get("sentiment_score", 0.0)

            follow_up = is_followup or last_context.get("follow_up_count", 0) >= 2
            follow_up_count = last_context.get("follow_up_count", 0) + 1 if follow_up else 0

            cls._SESSION_MEMORY[session_id] = {
                "urgency": urgency,
                "emotion": emotion,
                "sentiment_score": sentiment_score,
                "intent": last_context.get("intent"),
                "topic": last_context.get("topic"),
                "follow_up_count": follow_up_count
            }

            return {
                "urgency": urgency,
                "emotion": emotion,
                "sentiment_score": sentiment_score,
                "language": language,
                "follow_up": follow_up,
                "previous_intent": last_context.get("intent"),
                "previous_topic": last_context.get("topic"),
            }
