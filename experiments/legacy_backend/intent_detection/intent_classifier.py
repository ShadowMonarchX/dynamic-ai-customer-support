from typing import Dict, Any
from langdetect import detect, DetectorFactory
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama import OllamaLLM
import re

DetectorFactory.seed = 0

INTENTS = {
    "greeting",
    "identity",
    "service_query",
    "contact_request",
    "transactional",
    "unknown",
}

EMOTIONS = {"neutral", "frustrated", "angry", "urgent", "stressed"}

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
    "not working",
    "failed",
}

CONFIDENCE_THRESHOLD = 0.6

GREETING_REGEX = re.compile(r"^(hi+|hello+|hey+|yo+|hmm+|woo+|yay+)$", re.IGNORECASE)


class IntentClassifier:
    def __init__(self, model_name: str = "mistral"):
        self.llm = OllamaLLM(model=model_name, temperature=0.2, num_predict=128)
        self.parser = JsonOutputParser()
        self.prompt = PromptTemplate(
            template="""Return ONLY valid JSON.
intent: one of [{intents}]
emotion: one of [{emotions}]
urgency: low | medium | high
complexity: small | medium | big
confidence: 0.0-1.0
User message: "{message}"
""",
            input_variables=["message"],
            partial_variables={
                "intents": ", ".join(INTENTS),
                "emotions": ", ".join(EMOTIONS),
            },
        )

    def _apply_basic_rules(self, text: str) -> Dict[str, Any]:
        lowered = text.lower().strip()

        if GREETING_REGEX.fullmatch(lowered):
            return {
                "intent": "greeting",
                "emotion": "neutral",
                "urgency": "low",
                "complexity": "small",
                "confidence": 0.99,
            }

        if lowered.startswith("who is") or lowered.startswith("what is"):
            return {
                "intent": "identity",
                "confidence": 0.9,
                "complexity": "small",
            }

        if any(w in lowered for w in {"service", "offer", "provide", "work", "do you"}):
            return {
                "intent": "service_query",
                "confidence": 0.85,
                "complexity": "medium",
            }

        if any(w in lowered for w in {"contact", "email", "call"}):
            return {
                "intent": "contact_request",
                "confidence": 0.85,
                "complexity": "small",
            }

        if any(w in lowered for w in {"order", "refund", "cancel", "track"}):
            return {
                "intent": "transactional",
                "confidence": 0.85,
                "complexity": "medium",
            }

        return {}

    def _extract_human_features(self, text: str) -> Dict[str, Any]:
        lowered = text.lower()
        try:
            language = detect(text)
        except Exception:
            language = "unknown"

        return {
            "language": language,
            "emotion": (
                "frustrated"
                if any(w in lowered for w in FRUSTRATION_KEYWORDS)
                else "neutral"
            ),
            "urgency": "high" if any(w in lowered for w in URGENT_KEYWORDS) else "low",
            "sentiment_score": (
                -0.8 if any(w in lowered for w in FRUSTRATION_KEYWORDS) else 0.0
            ),
        }

    def classify(self, message: str) -> Dict[str, Any]:
        result = {
            "intent": "unknown",
            "emotion": "neutral",
            "urgency": "low",
            "complexity": "small",
            "language": "unknown",
            "sentiment_score": 0.0,
            "confidence": 0.0,
        }

        try:
            if not message or not message.strip():
                return result

            result.update(self._apply_basic_rules(message))
            result.update(self._extract_human_features(message))

            if result.get("intent") == "unknown":
                try:
                    chain = self.prompt | self.llm | self.parser
                    ai_out = chain.invoke({"message": message})
                    if ai_out.get("confidence", 0.0) >= CONFIDENCE_THRESHOLD:
                        result.update(ai_out)
                except Exception:
                    pass

            if result.get("confidence", 0.0) < CONFIDENCE_THRESHOLD:
                result["intent"] = "unknown"

            if result["intent"] not in INTENTS:
                result["intent"] = "unknown"

            return result
        except Exception:
            return result
