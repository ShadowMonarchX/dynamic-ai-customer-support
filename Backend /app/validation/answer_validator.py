import threading
import logging
from typing import Dict, Any, List
from langchain_core.runnables import Runnable # type: ignore

# Setup basic logging to track validation failures
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INTENT_RULES = {
    "greeting": {"min_len": 5, "max_len": 40, "must_have_question": True, "tone": "friendly"},
    "faq": {"min_len": 20, "max_len": 120, "must_have_question": True, "tone": "neutral"},
    "transactional": {"min_len": 15, "max_len": 100, "must_have_question": False, "tone": "data"},
    "big_issue": {"min_len": 60, "max_len": 300, "must_have_question": True, "tone": "empathetic"},
    "account_support": {"min_len": 40, "max_len": 200, "must_have_question": True, "tone": "empathetic"},
    "unknown": {"min_len": 10, "max_len": 120, "must_have_question": True, "tone": "neutral"},
}

class AnswerValidator(Runnable):
    def __init__(self):
        # 1. Thread Handling: Initialize a lock for concurrent safety
        self._lock = threading.Lock()

    def invoke(self, inputs: Dict[str, Any], config: Any = None) -> Dict[str, Any]:
        # 2. Thread Safety: Ensure only one thread executes validation logic at a time
        with self._lock:
            try:
                # 3. Error Handling: Securely extract inputs with defaults
                answer = str(inputs.get("answer", "")).strip()
                intent = inputs.get("intent", "unknown")
                
                if not answer:
                    return self._generate_error_response("Empty answer provided", answer)

                issues = []
                rules = INTENT_RULES.get(intent, INTENT_RULES["unknown"])
                length = len(answer)

                # Validation Logic
                if length < rules["min_len"]:
                    issues.append(f"Answer too short ({length} chars)")
                if length > rules["max_len"]:
                    issues.append(f"Answer too long ({length} chars)")
                if rules["must_have_question"] and "?" not in answer:
                    issues.append("Missing follow-up question")

                # Contextual Rules
                if intent in {"big_issue", "account_support"} and not self._has_empathy(answer):
                    issues.append("Missing empathy for sensitive issue")
                
                if intent == "transactional" and self._has_empathy(answer):
                    issues.append("Unnecessary emotion in transactional reply")

                if answer.lower() in {"i don't know", "not sure", "cannot help"}:
                    issues.append("Low-confidence response phrases detected")

                confidence = self._estimate_confidence(len(issues))

                return {
                    "valid": len(issues) == 0,
                    "confidence": confidence,
                    "issues": issues,
                    "answer": answer,
                    "status": "success"
                }

            except Exception as e:
                # 4. Error Resilience: Catch unexpected logic errors without crashing the main loop
                logger.error(f"Validation Error: {str(e)}")
                return self._generate_error_response(f"System error: {str(e)}", inputs.get("answer", ""))

    def _has_empathy(self, text: str) -> bool:
        empathy_phrases = ["i understand", "i’m sorry", "i know this is", "i can imagine"]
        lowered = text.lower()
        return any(phrase in lowered for phrase in empathy_phrases)

    def _estimate_confidence(self, issue_count: int) -> float:
        mapping = {0: 0.95, 1: 0.7, 2: 0.45}
        return mapping.get(issue_count, 0.2)

    def _generate_error_response(self, error_msg: str, original_answer: str) -> Dict[str, Any]:
        return {
            "valid": False,
            "confidence": 0.0,
            "issues": [error_msg],
            "answer": original_answer,
            "status": "error"
        }