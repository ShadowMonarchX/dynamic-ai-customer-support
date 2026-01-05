# from typing import Dict, Any
# from langchain_core.runnables import Runnable

# INTENT_RULES = {
#     "greeting": {"min_len": 3, "max_len": 30},
#     "faq": {"min_len": 15, "max_len": 120},
#     "services": {"min_len": 15, "max_len": 120},
#     "transactional": {"min_len": 15, "max_len": 100},
#     "big_issue": {"min_len": 40, "max_len": 250},
#     "unknown": {"min_len": 10, "max_len": 120},
# }


# class AnswerValidator(Runnable):
#     def __init__(self):
#         pass

#     def invoke(self, inputs: Dict[str, Any], config: Any = None) -> Dict[str, Any]:
#         try:
#             answer = str(inputs.get("answer", "")).strip()
#             intent = inputs.get("intent", "unknown")
#             similarity = float(inputs.get("similarity", 0.0))
#             context = inputs.get("context", "")

#             if not answer:
#                 return self._fallback("empty_answer", answer)

#             if not context:
#                 return self._fallback("empty_context", answer)

#             rules = INTENT_RULES.get(intent, INTENT_RULES["unknown"])
#             issues = []

#             length = len(answer)
#             if length < rules["min_len"]:
#                 issues.append("too_short")
#             if length > rules["max_len"]:
#                 issues.append("too_long")

#             if self._contains_guessing(answer):
#                 issues.append("guessing_language")

#             if similarity < 0.25:
#                 issues.append("low_similarity")

#             confidence = self._calculate_confidence(issues, similarity)

#             if confidence < 0.3:
#                 return self._fallback(
#                     "low_confidence",
#                     "I’m not fully sure. Could you please clarify your question?",
#                 )

#             return {
#                 "valid": confidence >= 0.6,
#                 "confidence": round(confidence, 2),
#                 "issues": issues,
#                 "trust_message": self._generate_trust_message(confidence),
#                 "answer": answer,
#                 "status": "success",
#             }

#         except Exception:
#             return self._fallback("validation_error", inputs.get("answer", ""))

#     def _calculate_confidence(self, issues: list, similarity: float) -> float:
#         score = similarity
#         penalties = {
#             "too_short": 0.1,
#             "too_long": 0.05,
#             "guessing_language": 0.2,
#             "low_similarity": 0.25,
#         }
#         for issue in issues:
#             score -= penalties.get(issue, 0.0)
#         return max(0.0, min(score, 1.0))

#     def _contains_guessing(self, text: str) -> bool:
#         phrases = {
#             "maybe",
#             "probably",
#             "i think",
#             "might be",
#             "not sure",
#             "could be",
#             "guess",
#         }
#         lowered = text.lower()
#         return any(p in lowered for p in phrases)

#     def _generate_trust_message(self, confidence: float) -> str:
#         if confidence >= 0.8:
#             return "High confidence answer."
#         if confidence >= 0.5:
#             return "Moderate confidence answer."
#         return "Low confidence answer."

#     def _fallback(self, reason: str, answer: str) -> Dict[str, Any]:
#         return {
#             "valid": False,
#             "confidence": 0.0,
#             "issues": [reason],
#             "trust_message": "I’m not fully sure. Please clarify your question.",
#             "answer": answer,
#             "status": "fallback",
#         }


from typing import Dict, Any
from langchain_core.runnables import Runnable

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
        pass

    def invoke(self, inputs: Dict[str, Any], config: Any = None) -> Dict[str, Any]:
        try:
            answer = str(inputs.get("answer", "")).strip()
            intent = inputs.get("intent", "unknown")
            similarity = float(inputs.get("similarity", 0.0))
            context = inputs.get("context", "")

            if not answer:
                return self._fallback("empty_answer", answer)

            if not context:
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
                    "I’m not fully sure. Please clarify your question?",
                )

            return {
                "valid": confidence >= 0.6,
                "confidence": round(confidence, 2),
                "issues": issues,
                "trust_message": self._generate_trust_message(confidence),
                "answer": answer,
                "status": "success",
                # ✅ Added missing keys with default scores
                "relevance": round(confidence, 2),
                "clarity": round(confidence, 2),
                "consistency": round(confidence, 2),
                "completeness": round(confidence, 2),
                "final_score": round(confidence, 2),
            }

        except Exception:
            return self._fallback("validation_error", inputs.get("answer", ""))

    def _calculate_confidence(self, issues: list, similarity: float) -> float:
        score = similarity
        penalties = {
            "too_short": 0.1,
            "too_long": 0.05,
            "guessing_language": 0.2,
            "low_similarity": 0.25,
        }
        for issue in issues:
            score -= penalties.get(issue, 0.0)
        return max(0.0, min(score, 1.0))

    def _contains_guessing(self, text: str) -> bool:
        phrases = {
            "maybe",
            "probably",
            "i think",
            "might be",
            "not sure",
            "could be",
            "guess",
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
            "status": "fallback",
            # ✅ Added all missing keys for fallback too
            "relevance": 0.0,
            "clarity": 0.0,
            "consistency": 0.0,
            "completeness": 0.0,
            "final_score": 0.0,
        }
