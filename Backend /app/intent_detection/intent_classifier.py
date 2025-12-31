import threading
from typing import Dict, Any
from langdetect import detect, DetectorFactory  # type: ignore
from langchain_core.prompts import PromptTemplate  # type: ignore
from langchain_core.output_parsers import JsonOutputParser  # type: ignore
from langchain_community.llms import Ollama  # type: ignore

DetectorFactory.seed = 0

# Narrowed intent classes
INTENTS = {"greeting", "identity_lookup", "service_query", "contact_request", "transactional", "unknown"}
EMOTIONS = {"neutral", "frustrated", "angry", "urgent", "stressed"}
URGENT_KEYWORDS = {"now", "urgent", "asap", "immediately", "today", "tomorrow", "right away"}
FRUSTRATION_KEYWORDS = {"angry", "frustrated", "annoyed", "ridiculous", "worst", "not working", "failed"}


class IntentClassifier:
    def __init__(self, model_name: str = "mistral"):
        self._lock = threading.Lock()
        self.llm = Ollama(model=model_name, temperature=0.2, num_predict=128)
        self.parser = JsonOutputParser()
        self.prompt = PromptTemplate(
            template="""Return ONLY valid JSON.
                        Fields:
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
                "emotions": ", ".join(EMOTIONS)
            }
        )

    def _apply_basic_rules(self, text: str) -> Dict[str, Any]:
        lowered = text.lower().strip()
        result = {}

        # Greeting detection
        if lowered in {"hi", "hello", "hey"}:
            result.update({"intent": "greeting", "emotion": "neutral", "urgency": "low", "complexity": "small", "confidence": 0.95})
        
        # Identity lookup
        elif any(word in lowered for word in {"who", "what", "whose"}) :
            result.update({"intent": "identity_lookup", "confidence": 0.9, "complexity": "small"})

        # Service query
        elif any(word in lowered for word in {"how", "help", "support", "guide"}):
            result.update({"intent": "service_query", "confidence": 0.85, "complexity": "medium"})

        # Contact requests / transactional
        elif any(word in lowered for word in {"contact", "call", "email", "refund", "order", "cancel", "track"}):
            result.update({"intent": "contact_request", "confidence": 0.85, "complexity": "medium"})

        # Frustration detection bumps urgency
        if any(w in lowered for w in FRUSTRATION_KEYWORDS):
            result["emotion"] = "frustrated"
            result["urgency"] = "high"
        
        if any(w in lowered for w in URGENT_KEYWORDS):
            result["urgency"] = "high"

        return result

    def _extract_human_features(self, text: str) -> Dict[str, Any]:
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
                "sentiment_score": 0.0,
                "confidence": 0.0
            }

            if not message or not message.strip():
                return result

            # 1️⃣ Apply rule-based heuristics first
            result.update(self._apply_basic_rules(message))

            # 2️⃣ Extract human-centric features
            human_features = self._extract_human_features(message)
            result.update(human_features)

            # 3️⃣ Use LLM fallback if intent still unknown
            if result.get("intent") == "unknown":
                try:
                    chain = self.prompt | self.llm | self.parser
                    ai_out = chain.invoke({"message": message})
                    result.update(ai_out)
                except Exception:
                    pass  # fail-safe

            # 4️⃣ Normalize outputs
            if result["intent"] not in INTENTS:
                result["intent"] = "unknown"
            if result["emotion"] not in EMOTIONS:
                result["emotion"] = "neutral"
            if result["urgency"] not in {"low", "medium", "high"}:
                result["urgency"] = "low"
            if result["complexity"] not in {"small", "medium", "big"}:
                result["complexity"] = "small"
            if "confidence" not in result:
                result["confidence"] = 0.0

            return result
