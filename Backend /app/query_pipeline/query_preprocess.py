import re
from typing import Dict, Any
from langchain_core.runnables import Runnable  # type: ignore

URGENT_KEYWORDS = [
    "now", "urgent", "asap", "immediately", "today", "tomorrow", "right away"
]

FRUSTRATION_KEYWORDS = [
    "angry", "frustrated", "annoyed", "ridiculous", "worst",
    "not working", "failed", "again", "third time"
]

class QueryPreprocessor(Runnable):
    def invoke(self, query: str) -> Dict[str, Any]:
        if not isinstance(query, str) or not query.strip():
            return {
                "clean_text": "",
                "urgency": "low",
                "emotion": "neutral"
            }

        lowered = query.lower()
        clean_text = re.sub(r"[^a-z0-9\s]", "", lowered).strip()

        urgency = (
            "high"
            if any(word in lowered for word in URGENT_KEYWORDS)
            else "low"
        )

        emotion = (
            "frustrated"
            if any(word in lowered for word in FRUSTRATION_KEYWORDS)
            else "neutral"
        )

        return {
            "clean_text": clean_text,
            "urgency": urgency,
            "emotion": emotion
        }
