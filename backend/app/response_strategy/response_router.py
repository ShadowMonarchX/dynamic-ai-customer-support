from typing import Dict
from .response_strategy import (
    GreetingStrategy,
    EmotionStrategy,
    BigIssueStrategy,
    TransactionalStrategy,
    FAQStrategy,
    FallbackStrategy
)


class ResponseStrategyRouter:
    """
    Router selects the appropriate response strategy
    based on features extracted from user query.
    """

    def __init__(self):
        # Priority order
        self.strategies = [
            GreetingStrategy(),
            EmotionStrategy(),
            BigIssueStrategy(),
            TransactionalStrategy(),
            FAQStrategy(),
            FallbackStrategy()
        ]

    def select(self, features: Dict) -> str:
        """
        Returns the system prompt string for the most appropriate strategy.
        """
        for strategy in self.strategies:
            if strategy.is_applicable(features):
                return strategy.system_prompt(features)


# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    router = ResponseStrategyRouter()

    features_example = {
        "intent": "identity_lookup",
        "emotion": "neutral",
        "urgency": "normal",
        "is_greeting": False,
        "complexity": "simple"
    }

    prompt = router.select(features_example)
    print("Selected System Prompt:\n", prompt)
