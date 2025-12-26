from typing import Dict, Any
from langchain.schema import Runnable


class AnswerValidator(Runnable):
    """
    LangChain-style answer validator:
    - Length check
    - Low-confidence detection
    - Context relevance check
    - Confidence scoring
    """

    def __init__(self, min_length: int = 20):
        self.min_length = min_length

    def invoke(self, inputs: Dict[str, str]) -> Dict[str, Any]:
        answer = (inputs.get("answer") or "").strip()
        context = (inputs.get("context") or "").lower()

        issues = []

        if len(answer) < self.min_length:
            issues.append("Answer is incomplete")

        if answer.lower() in {"i don't know", "not sure"}:
            issues.append("Low confidence answer")

        if answer:
            answer_tokens = set(answer.lower().split())
            context_tokens = set(context.split())
            if not answer_tokens & context_tokens:
                issues.append("Answer may be unrelated to context")

        confidence = self._estimate_confidence(len(issues))

        return {
            "valid": len(issues) == 0,
            "confidence": confidence,
            "issues": issues,
            "answer": answer,
        }

    def _estimate_confidence(self, issue_count: int) -> float:
        if issue_count == 0:
            return 0.9
        if issue_count == 1:
            return 0.6
        return 0.3
