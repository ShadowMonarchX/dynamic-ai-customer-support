# import threading
# import logging
# from typing import Dict, Any
# from langchain_core.runnables import Runnable

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# INTENT_RULES = {
#     "greeting": {"min_len": 3, "max_len": 30, "require_question": False},
#     "faq": {"min_len": 15, "max_len": 120, "require_question": False},
#     "services": {"min_len": 15, "max_len": 120, "require_question": False},
#     "transactional": {"min_len": 15, "max_len": 100, "require_question": False},
#     "big_issue": {"min_len": 40, "max_len": 250, "require_question": False},
#     "unknown": {"min_len": 10, "max_len": 120, "require_question": False},
# }


# class AnswerValidator(Runnable):
#     def __init__(self):
#         self._lock = threading.Lock()

#     def invoke(self, inputs: Dict[str, Any], config: Any = None) -> Dict[str, Any]:
#         with self._lock:
#             try:
#                 answer = str(inputs.get("answer", "")).strip()
#                 intent = inputs.get("intent", "unknown")
#                 similarity = float(inputs.get("similarity", 0.0))

#                 if not answer:
#                     return self._fallback("Empty answer", answer)

#                 rules = INTENT_RULES.get(intent, INTENT_RULES["unknown"])
#                 issues = []

               
#                 length = len(answer)
#                 if length < rules["min_len"]:
#                     issues.append("too_short")
#                 if length > rules["max_len"]:
#                     issues.append("too_long")

               
#                 if self._contains_guessing(answer):
#                     issues.append("guessing_language")

            
#                 confidence = self._calculate_confidence(issues, similarity)

            
#                 trust_message = self._generate_trust_message(confidence, issues)

    
#                 if similarity < 0.25 or confidence < 0.2:
#                     return self._fallback(
#                         "Low semantic match",
#                         "I’m not fully sure about this. Could you please clarify your question?"
#                     )

#                 return {
#                     "valid": confidence >= 0.6,
#                     "confidence": round(confidence, 2),
#                     "issues": issues,
#                     "trust_message": trust_message,
#                     "answer": answer,
#                     "status": "success"
#                 }

#             except Exception as e:
#                 logger.error(f"Answer validation error: {e}")
#                 return self._fallback("Validation failure", inputs.get("answer", ""))

#     def _calculate_confidence(self, issues, similarity) -> float:
#         """Weighted confidence calculation with issue penalties"""
#         score = similarity
#         if "too_short" in issues:
#             score -= 0.1
#         if "too_long" in issues:
#             score -= 0.05
#         if "guessing_language" in issues:
#             score -= 0.2

#         return max(0.0, min(score, 1.0))

#     def _contains_guessing(self, text: str) -> bool:
#         bad_phrases = [
#             "maybe", "probably", "i think", "might be",
#             "not sure", "could be", "guess"
#         ]
#         lowered = text.lower()
#         return any(p in lowered for p in bad_phrases)

#     def _generate_trust_message(self, confidence: float, issues: list) -> str:
#         """Provide user-friendly trust feedback"""
#         if confidence >= 0.8:
#             return "High confidence: Answer is reliable."
#         elif confidence >= 0.5:
#             return "Moderate confidence: Answer seems correct but may need clarification."
#         else:
#             return "Low confidence: Answer may be inaccurate; consider rephrasing your question."

#     def _fallback(self, reason: str, answer: str) -> Dict[str, Any]:
#         return {
#             "valid": False,
#             "confidence": 0.0,
#             "issues": [reason],
#             "trust_message": "I’m not sure about this. Please clarify your question.",
#             "answer": answer,
#             "status": "fallback"
#         }

import threading
import logging
from typing import Dict, Any
from langchain_core.runnables import Runnable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INTENT_RULES = {
    "greeting": {"min_len": 3, "max_len": 30},
    "faq": {"min_len": 15, "max_len": 120},
    "services": {"min_len": 15, "max_len": 120},
    "transactional": {"min_len": 15, "max_len": 100},
    "big_issue": {"min_len": 40, "max_len": 250},
    "unknown": {"min_len": 10, "max_len": 120},
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
                context = inputs.get("context", "")

                if not answer:
                    logger.warning("Empty answer detected")
                    return self._fallback("empty_answer", answer)

                if not context:
                    logger.warning("Empty context detected")
                    return self._fallback("empty_context", answer)

                rules = INTENT_RULES.get(intent, INTENT_RULES["unknown"])
                issues = []

                length = len(answer)
                if length < rules["min_len"]:
                    issues.append("too_short")
                if length > rules["max_len"]:
                    issues.append("too_long")

                if self._contains_guessing(answer):
                    issues.append("guessing_language")

                if similarity < 0.25:
                    issues.append("low_similarity")

                confidence = self._calculate_confidence(issues, similarity)

                if confidence < 0.3:
                    return self._fallback(
                        "low_confidence",
                        "I’m not fully sure. Could you please clarify your question?"
                    )

                return {
                    "valid": confidence >= 0.6,
                    "confidence": round(confidence, 2),
                    "issues": issues,
                    "trust_message": self._generate_trust_message(confidence),
                    "answer": answer,
                    "status": "success"
                }

            except Exception as e:
                logger.error(f"Validation error: {e}")
                return self._fallback("validation_error", inputs.get("answer", ""))

    def _calculate_confidence(self, issues: list, similarity: float) -> float:
        score = similarity
        penalties = {
            "too_short": 0.1,
            "too_long": 0.05,
            "guessing_language": 0.2,
            "low_similarity": 0.25
        }
        for issue in issues:
            score -= penalties.get(issue, 0.0)
        return max(0.0, min(score, 1.0))

    def _contains_guessing(self, text: str) -> bool:
        phrases = {
            "maybe", "probably", "i think", "might be",
            "not sure", "could be", "guess"
        }
        lowered = text.lower()
        return any(p in lowered for p in phrases)

    def _generate_trust_message(self, confidence: float) -> str:
        if confidence >= 0.8:
            return "High confidence answer."
        if confidence >= 0.5:
            return "Moderate confidence answer."
        return "Low confidence answer."

    def _fallback(self, reason: str, answer: str) -> Dict[str, Any]:
        return {
            "valid": False,
            "confidence": 0.0,
            "issues": [reason],
            "trust_message": "I’m not fully sure. Please clarify your question.",
            "answer": answer,
            "status": "fallback"
        }
