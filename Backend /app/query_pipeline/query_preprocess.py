import re
import threading
import logging
from typing import Dict, Any
from langdetect import detect, DetectorFactory
import spacy

DetectorFactory.seed = 0
logger = logging.getLogger(__name__)
nlp = spacy.load("en_core_web_sm")

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

# Patterns for small talk / casual exclamations
SMALL_TALK_PATTERNS = [
    r"h+i+",  # hi, hii, hiiii
    r"hello+",  # hello, helloo, hellooo
    r"hey+",  # hey, heyy
    r"wo+o+",  # woo, wooo
    r"yay+",  # yayyy
    r"yo+",  # yo, yoo
    r"hmm+",  # hmm, hmmm
    r"hooray+",  # hooray, hoorayyy
    r"so what",  # casual
    r"okay+",  # ok, okay, okey
]
COMPILED_SMALL_TALK = [re.compile(p, re.IGNORECASE) for p in SMALL_TALK_PATTERNS]


def is_small_talk(text: str) -> bool:
    text = text.strip().lower()
    return any(p.fullmatch(text) for p in COMPILED_SMALL_TALK)


class QueryPreprocessor:
    def __init__(self):
        self._lock = threading.Lock()
        self.clean_regex = re.compile(r"[^a-z0-9\s\?!]")

    def invoke(self, query: str) -> Dict[str, Any]:
        with self._lock:
            try:
                if not isinstance(query, str) or not query.strip():
                    return self._default_response()
                lowered = query.lower().strip()

                # Detect small talk
                if is_small_talk(lowered):
                    return {
                        "clean_text": lowered,
                        "urgency": "low",
                        "emotion": "neutral",
                        "sentiment_score": 0.0,
                        "language": "unknown",
                        "intent": "small_talk",
                        "named_entities": [],
                        "question_depth": 0,
                    }

                clean_text = self.clean_regex.sub("", lowered).strip()
                if not clean_text:
                    return self._default_response()

                named_entities = [ent.text for ent in nlp(query).ents]
                is_urgent = any(word in lowered for word in URGENT_KEYWORDS)
                is_frustrated = any(word in lowered for word in FRUSTRATION_KEYWORDS)

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
            except Exception:
                return self._default_response()

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
