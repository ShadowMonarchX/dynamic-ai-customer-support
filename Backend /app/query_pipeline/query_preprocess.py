# query_preprocess.py
# (Query Enhancement Layer)
# Purpose
#
# Improve user queries before searching the knowledge base.
#
# What Happens Here
#
# Removes unnecessary or emotional words
# Adds missing business context
# Converts human language into searchable meaning
#
# Example
#
# User query:
# > “Order not coming yet”
#
# Enhanced internal query:
# > “Order delivery delay reasons and expected delivery time”
#
# This step ensures better retrieval accuracy.

import re
import threading
from typing import Dict, Any
from langdetect import detect, DetectorFactory  # type: ignore

DetectorFactory.seed = 0

URGENT_KEYWORDS = {"now", "urgent", "asap", "immediately", "today", "tomorrow", "right away"}
FRUSTRATION_KEYWORDS = {"angry", "frustrated", "annoyed", "ridiculous", "worst", "failed", "again"}

class QueryPreprocessor:
    def __init__(self):
        self._lock = threading.Lock()
        self.clean_regex = re.compile(r"[^a-z0-9\s]")

    def invoke(self, query: str) -> Dict[str, Any]:
        with self._lock:
            try:
                if not isinstance(query, str) or not query.strip():
                    return self._default_response()

                lowered = query.lower()
                clean_text = self.clean_regex.sub("", lowered).strip()

                is_urgent = any(word in lowered for word in URGENT_KEYWORDS)
                is_frustrated = any(word in lowered for word in FRUSTRATION_KEYWORDS)

                try:
                    language = detect(query)
                except:
                    language = "unknown"

                return {
                    "clean_text": clean_text,
                    "urgency": "high" if is_urgent else "low",
                    "emotion": "frustrated" if is_frustrated else "neutral",
                    "sentiment_score": -0.8 if is_frustrated else 0.0,
                    "language": language,
                    "intent": self._detect_basic_intent(lowered)
                }
            except Exception as e:
                raise RuntimeError(f"Query Preprocessing Failed: {e}")

    def _detect_basic_intent(self, text: str) -> str:
        if "refund" in text or "money back" in text:
            return "refund"
        if "help" in text or "how to" in text:
            return "faq"
        return "general"

    def _default_response(self) -> Dict[str, Any]:
        return {
            "clean_text": "",
            "urgency": "low",
            "emotion": "neutral",
            "sentiment_score": 0.0,
            "language": "unknown",
            "intent": "general"
        }
