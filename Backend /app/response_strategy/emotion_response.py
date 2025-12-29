class EmotionResponseStrategy:
    def generate_response(self, emotion: str) -> str:
        if emotion in {"frustrated", "angry", "stressed"}:
            return (
                "I’m really sorry for the frustration this has caused.\n"
                "I’m here with you and we’ll resolve this as quickly as possible."
            )

        return "I’m here to help. Let’s work through this together."
