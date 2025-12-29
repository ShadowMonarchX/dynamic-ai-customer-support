# from typing import Dict, Any
# from langchain_core.runnables import Runnable # type: ignore

# class AnswerValidator(Runnable):
#     def __init__(self, min_length: int = 20):
#         self.min_length = min_length

#     def invoke(self, inputs: Dict[str, str]) -> Dict[str, Any]:
#         answer = (inputs.get("answer") or "").strip()
#         context = (inputs.get("context") or "").lower()
#         issues = []

#         if len(answer) < self.min_length:
#             issues.append("Answer is incomplete")
#         if answer.lower() in {"i don't know", "not sure"}:
#             issues.append("Low confidence answer")
#         if answer:
#             answer_tokens = set(answer.lower().split())
#             context_tokens = set(context.split())
#             if not answer_tokens & context_tokens:
#                 issues.append("Answer may be unrelated to context")

#         confidence = self._estimate_confidence(len(issues))
#         return {
#             "valid": len(issues) == 0,
#             "confidence": confidence,
#             "issues": issues,
#             "answer": answer,
#         }

#     def _estimate_confidence(self, issue_count: int) -> float:
#         if issue_count == 0:
#             return 0.9
#         if issue_count == 1:
#             return 0.6
#         return 0.3


from typing import Dict, Any
from langchain_core.runnables import Runnable  # type: ignore


INTENT_RULES = {
    "greeting": {
        "min_len": 5,
        "max_len": 40,
        "must_have_question": True,
        "tone": "friendly",
    },
    "faq": {
        "min_len": 20,
        "max_len": 120,
        "must_have_question": True,
        "tone": "neutral",
    },
    "transactional": {
        "min_len": 15,
        "max_len": 100,
        "must_have_question": False,
        "tone": "data",
    },
    "big_issue": {
        "min_len": 60,
        "max_len": 300,
        "must_have_question": True,
        "tone": "empathetic",
    },
    "account_support": {
        "min_len": 40,
        "max_len": 200,
        "must_have_question": True,
        "tone": "empathetic",
    },
    "unknown": {
        "min_len": 10,
        "max_len": 120,
        "must_have_question": True,
        "tone": "neutral",
    },
}


class AnswerValidator(Runnable):
    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        answer = (inputs.get("answer") or "").strip()
        intent = inputs.get("intent", "unknown")
        emotion = inputs.get("emotion", "neutral")

        issues = []

        rules = INTENT_RULES.get(intent, INTENT_RULES["unknown"])

        length = len(answer)

        if length < rules["min_len"]:
            issues.append("Answer too short for intent")

        if length > rules["max_len"]:
            issues.append("Answer too long for intent")

        if rules["must_have_question"] and "?" not in answer:
            issues.append("Missing next-step question")

        if intent in {"big_issue", "account_support"}:
            if not self._has_empathy(answer):
                issues.append("Missing empathy for high-impact issue")

        if intent == "transactional":
            if self._has_empathy(answer):
                issues.append("Unnecessary emotional language in transactional response")

        if answer.lower() in {"i don't know", "not sure", "cannot help"}:
            issues.append("Low-confidence response")

        confidence = self._estimate_confidence(len(issues))

        return {
            "valid": len(issues) == 0,
            "confidence": confidence,
            "issues": issues,
            "answer": answer,
        }

    def _has_empathy(self, text: str) -> bool:
        empathy_phrases = [
            "i understand",
            "i’m sorry",
            "i am sorry",
            "that must be",
            "i know this is",
            "i can imagine",
        ]
        lowered = text.lower()
        return any(phrase in lowered for phrase in empathy_phrases)

    def _estimate_confidence(self, issue_count: int) -> float:
        if issue_count == 0:
            return 0.95
        if issue_count == 1:
            return 0.7
        if issue_count == 2:
            return 0.45
        return 0.2
