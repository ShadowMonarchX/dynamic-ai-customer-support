import threading
import logging
from typing import Dict, Any
from langchain_core.runnables import Runnable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INTENT_RULES = {
    "greeting": {"min_len": 3, "max_len": 30, "require_question": False},
    "faq": {"min_len": 15, "max_len": 120, "require_question": False},
    "services": {"min_len": 15, "max_len": 120, "require_question": False},
    "transactional": {"min_len": 15, "max_len": 100, "require_question": False},
    "big_issue": {"min_len": 40, "max_len": 250, "require_question": False},
    "unknown": {"min_len": 10, "max_len": 120, "require_question": False},
}


class AnswerValidator(Runnable):
    def __init__(self):
        self._lock = threading.Lock()

    def invoke(self, inputs: Dict[str, Any], config: Any = None) -> Dict[str, Any]:
        with self._lock:
            try:
                answer = str(inputs.get("answer", "")).strip()
                intent = inputs.get("intent", "unknown")
                similarity = float(inputs.get("similarity", 0.0))

                if not answer:
                    return self._fallback("Empty answer", answer)

                rules = INTENT_RULES.get(intent, INTENT_RULES["unknown"])
                issues = []

               
                length = len(answer)
                if length < rules["min_len"]:
                    issues.append("too_short")
                if length > rules["max_len"]:
                    issues.append("too_long")

               
                if self._contains_guessing(answer):
                    issues.append("guessing_language")

            
                confidence = self._calculate_confidence(issues, similarity)

            
                trust_message = self._generate_trust_message(confidence, issues)

    
                if similarity < 0.25 or confidence < 0.2:
                    return self._fallback(
                        "Low semantic match",
                        "I’m not fully sure about this. Could you please clarify your question?"
                    )

                return {
                    "valid": confidence >= 0.6,
                    "confidence": round(confidence, 2),
                    "issues": issues,
                    "trust_message": trust_message,
                    "answer": answer,
                    "status": "success"
                }

            except Exception as e:
                logger.error(f"Answer validation error: {e}")
                return self._fallback("Validation failure", inputs.get("answer", ""))

    def _calculate_confidence(self, issues, similarity) -> float:
        """Weighted confidence calculation with issue penalties"""
        score = similarity
        if "too_short" in issues:
            score -= 0.1
        if "too_long" in issues:
            score -= 0.05
        if "guessing_language" in issues:
            score -= 0.2

        return max(0.0, min(score, 1.0))

    def _contains_guessing(self, text: str) -> bool:
        bad_phrases = [
            "maybe", "probably", "i think", "might be",
            "not sure", "could be", "guess"
        ]
        lowered = text.lower()
        return any(p in lowered for p in bad_phrases)

    def _generate_trust_message(self, confidence: float, issues: list) -> str:
        """Provide user-friendly trust feedback"""
        if confidence >= 0.8:
            return "High confidence: Answer is reliable."
        elif confidence >= 0.5:
            return "Moderate confidence: Answer seems correct but may need clarification."
        else:
            return "Low confidence: Answer may be inaccurate; consider rephrasing your question."

    def _fallback(self, reason: str, answer: str) -> Dict[str, Any]:
        return {
            "valid": False,
            "confidence": 0.0,
            "issues": [reason],
            "trust_message": "I’m not sure about this. Please clarify your question.",
            "answer": answer,
            "status": "fallback"
        }
