import threading
from typing import Dict, Any
from langdetect import detect, DetectorFactory
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.llms import Ollama

DetectorFactory.seed = 0

INTENTS = {"greeting", "faq", "transactional", "big_issue", "account_support", "unknown"}
EMOTIONS = {"neutral", "frustrated", "angry", "urgent", "stressed"}
URGENT_KEYWORDS = {"now", "urgent", "asap", "immediately", "today", "tomorrow", "right away"}
FRUSTRATION_KEYWORDS = {"angry", "frustrated", "annoyed", "ridiculous", "worst", "not working", "failed"}

class IntentClassifier:
    def __init__(self, model_name: str = "mistral"):
        self._lock = threading.Lock()
        self.llm = Ollama(
            model=model_name,
            temperature=0.2,      # NEVER 0 on MPS
            num_predict=128
        )
        self.parser = JsonOutputParser()
        self.prompt = PromptTemplate(
            template="""Return ONLY valid JSON.
Fields:
intent: one of [{intents}]
emotion: one of [{emotions}]
urgency: low | medium | high
complexity: small | medium | big
User message: "{message}"
""",
            input_variables=["message"],
            partial_variables={
                "intents": ", ".join(INTENTS),
                "emotions": ", ".join(EMOTIONS)
            }
        )

    def _apply_basic_rules(self, text: str) -> Dict[str, Any]:
        lowered = text.lower().strip()

        # Greeting detection
        if lowered in {"hi", "hey", "hello"}:
            return {
                "intent": "greeting",
                "emotion": "neutral",
                "urgency": "low",
                "complexity": "small"
            }

        # Transactional keywords
        if any(w in lowered for w in ["refund", "cancel", "return", "order", "track"]):
            return {
                "intent": "transactional",
                "emotion": "neutral",
                "urgency": "medium",
                "complexity": "medium"
            }

        # Frustration or account issues
        if any(w in lowered for w in FRUSTRATION_KEYWORDS):
            return {
                "intent": "account_support",
                "emotion": "frustrated",
                "urgency": "high",
                "complexity": "medium"
            }

        # Urgency detection
        if any(w in lowered for w in URGENT_KEYWORDS):
            return {"urgency": "high"}

        return {}

    def _extract_human_features(self, text: str) -> Dict[str, Any]:
        """
        Extract additional human-centric features: urgency, emotion, sentiment, language.
        """
        lowered = text.lower()
        is_urgent = any(word in lowered for word in URGENT_KEYWORDS)
        is_frustrated = any(word in lowered for word in FRUSTRATION_KEYWORDS)
        try:
            language = detect(text)
        except:
            language = "unknown"
        return {
            "urgency": "high" if is_urgent else "low",
            "emotion": "frustrated" if is_frustrated else "neutral",
            "sentiment_score": -0.8 if is_frustrated else 0.0,
            "language": language
        }

    def classify(self, message: str) -> Dict[str, Any]:
        with self._lock:
            result = {
                "intent": "unknown",
                "emotion": "neutral",
                "urgency": "low",
                "complexity": "small",
                "language": "unknown",
                "sentiment_score": 0.0
            }

            if not message or not message.strip():
                return result

            # Apply rule-based heuristics
            result.update(self._apply_basic_rules(message))

            # Extract human-centric features
            human_features = self._extract_human_features(message)
            result.update(human_features)

            # Use LLM if intent is still unknown
            if result["intent"] == "unknown":
                try:
                    chain = self.prompt | self.llm | self.parser
                    ai_out = chain.invoke({"message": message})
                    result.update(ai_out)
                except Exception:
                    # 🔒 HARD FAILSAFE
                    return result

            # Normalize outputs
            result["intent"] = result["intent"] if result["intent"] in INTENTS else "unknown"
            result["emotion"] = result["emotion"] if result["emotion"] in EMOTIONS else "neutral"
            result["urgency"] = result["urgency"] if result["urgency"] in {"low", "medium", "high"} else "low"
            result["complexity"] = result["complexity"] if result["complexity"] in {"small", "medium", "big"} else "small"

            return result
