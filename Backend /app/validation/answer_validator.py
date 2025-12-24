class AnswerValidator:
    def __init__(self, min_length=20):
        self.min_length = min_length

    def validate(self, answer, context):
        answer = (answer or "").strip()
        context = context.lower()

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
            "valid": not issues,
            "confidence": confidence,
            "issues": issues,
        }

    def _estimate_confidence(self, issue_count):
        return 0.9 if issue_count == 0 else 0.6 if issue_count == 1 else 0.3
