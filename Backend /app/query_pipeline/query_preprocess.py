import re
from typing import Dict, Any
from langchain_core.runnables import Runnable  # type: ignore
from langdetect import detect, DetectorFactory # type: ignore

# Ensure consistent language detection
DetectorFactory.seed = 0

# Keywords for urgency/emotion
URGENT_KEYWORDS = [
    "now", "urgent", "asap", "immediately", "today", "tomorrow", "right away"
]

FRUSTRATION_KEYWORDS = [
    "angry", "frustrated", "annoyed", "ridiculous", "worst",
    "not working", "failed", "again", "third time"
]

class QueryPreprocessor(Runnable):
    """
    Preprocesses user query and extracts human-level features:
    - Sentiment (simplified as frustration detection)
    - Urgency flag
    - Language
    - Cleaned text
    """

    def invoke(self, query: str) -> Dict[str, Any]:
        if not isinstance(query, str) or not query.strip():
            return {
                "clean_text": "",
                "urgency": "low",
                "emotion": "neutral",
                "language": "unknown",
                "sentiment_score": 0.0
            }

        lowered = query.lower()
        
        # Clean text
        clean_text = re.sub(r"[^a-z0-9\s]", "", lowered).strip()

        # Urgency detection
        urgency_flag = any(word in lowered for word in URGENT_KEYWORDS)
        urgency = "high" if urgency_flag else "low"

        # Emotion detection (simplified frustration detection)
        is_frustrated = any(word in lowered for word in FRUSTRATION_KEYWORDS)
        emotion = "frustrated" if is_frustrated else "neutral"

        # Sentiment score (simplified: frustrated → negative, neutral → zero)
        sentiment_score = -0.8 if is_frustrated else 0.0

        # Language detection
        try:
            language = detect(query)
        except:
            language = "unknown"

        return {
            "clean_text": clean_text,
            "urgency": urgency,
            "emotion": emotion,
            "sentiment_score": sentiment_score,
            "language": language
        }
