import re
import threading
import logging
from typing import Dict, Any
from langdetect import detect, DetectorFactory  # type: ignore

DetectorFactory.seed = 0
logger = logging.getLogger(__name__)

# -------- spaCy safe load --------
try:
    import spacy  # type: ignore

    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None
    logger.warning("spaCy model not loaded — NER disabled")

# -------- Keyword sets --------
URGENT_KEYWORDS = {
    "now",
    "urgent",
    "asap",
    "immediately",
    "today",
    "tomorrow",
    "right away",
}

FRUSTRATION_KEYWORDS = {
    "angry",
    "frustrated",
    "annoyed",
    "ridiculous",
    "worst",
    "failed",
    "again",
}

QUESTION_WORDS = {"who", "what", "how", "why", "when", "where"}

# -------- Small talk patterns --------
SMALL_TALK_PATTERNS = [
    r"\bhi+\b",
    r"\bhello+\b",
    r"\bhey+\b",
    r"\bwo+o+\b",
    r"\byay+\b",
    r"\byo+\b",
    r"\bhmm+\b",
    r"\bhooray+\b",
    r"\bok(ay)?+\b",
    r"\bso what\b",
]

COMPILED_SMALL_TALK = [re.compile(p, re.IGNORECASE) for p in SMALL_TALK_PATTERNS]


def is_small_talk(text: str) -> bool:
    return any(p.search(text) for p in COMPILED_SMALL_TALK)


class QueryPreprocessor:
    def __init__(self):
        self._lock = threading.Lock()
        self.clean_regex = re.compile(r"[^a-z0-9\s\?!]")

    def invoke(self, query: str) -> Dict[str, Any]:
        with self._lock:
            if not isinstance(query, str) or not query.strip():
                return self._default_response()

            lowered = query.lower().strip()

            # ---- Small talk handling ----
            if is_small_talk(lowered):
                return {
                    "clean_text": lowered,
                    "urgency": "low",
                    "emotion": "neutral",
                    "sentiment_score": 0.0,
                    "language": "unknown",
                    "intent": "greeting",  
                    "named_entities": [],
                    "question_depth": 0,
                }

            clean_text = self.clean_regex.sub("", lowered).strip()
            if not clean_text:
                return self._default_response()

            # ---- NER ----
            named_entities = []
            if nlp:
                named_entities = [ent.text for ent in nlp(query).ents]

            is_urgent = any(word in lowered for word in URGENT_KEYWORDS)
            is_frustrated = any(word in lowered for word in FRUSTRATION_KEYWORDS)

            # ---- Language detection (skip for short text) ----
            if len(clean_text.split()) <= 2:
                language = "unknown"
            else:
                try:
                    language = detect(query)
                except Exception:
                    language = "unknown"

            question_depth = 2 if any(qw in lowered for qw in QUESTION_WORDS) else 1

            return {
                "clean_text": clean_text,
                "urgency": "high" if is_urgent else "low",
                "emotion": "frustrated" if is_frustrated else "neutral",
                "sentiment_score": -0.8 if is_frustrated else 0.0,
                "language": language,
                "intent": self._detect_basic_intent(lowered, named_entities),
                "named_entities": named_entities,
                "question_depth": question_depth,
            }

    def _detect_basic_intent(self, text: str, entities: list) -> str:
        if "refund" in text or "money back" in text:
            return "refund"
        if "help" in text or "how to" in text:
            return "faq"
        if any(qw in text for qw in {"who", "what"}) and entities:
            return "identity"
        return "general"

    def _default_response(self) -> Dict[str, Any]:
        return {
            "clean_text": "",
            "urgency": "low",
            "emotion": "neutral",
            "sentiment_score": 0.0,
            "language": "unknown",
            "intent": "general",
            "named_entities": [],
            "question_depth": 0,
        }
