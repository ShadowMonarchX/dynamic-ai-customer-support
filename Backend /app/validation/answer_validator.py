class AnswerValidator:
    def __init__(self, min_length=20):
        self.min_length = min_length

    def validate(self, answer, context):
        issues = []

        if not answer or len(answer.strip()) < self.min_length:
            issues.append("Answer is incomplete")

        if answer.lower() in ["i don't know", "not sure"]:
            issues.append("Low confidence answer")

        if not any(word in answer.lower() for word in context.lower().split()[:20]):
            issues.append("Answer may be unrelated to context")

        confidence = self._estimate_confidence(issues)

        return {
            "valid": len(issues) == 0,
            "confidence": confidence,
            "issues": issues
        }

    def _estimate_confidence(self, issues):
        if not issues:
            return 0.9
        if len(issues) == 1:
            return 0.6
        return 0.3
