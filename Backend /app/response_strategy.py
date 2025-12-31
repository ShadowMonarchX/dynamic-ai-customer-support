# app/response_strategy.py

class ResponseStrategy:
    """
    Class to select system response strategy based on
    intent, emotion, urgency, and context.
    """

    def __init__(self):
        # Define strategy templates
        self.strategies = {
            "identity_safe": "Answer strictly based on verified identity knowledge. Do not speculate.",
            "knowledge_missing": "You don't have enough information. Ask the user for clarification politely.",
            "default": "Answer helpfully and professionally based on available knowledge.",
            "urgent_emotion": "Prioritize urgent, empathetic responses.",
            "calm_emotion": "Provide detailed, informative responses at a calm pace."
        }

        # Mapping: intent + emotion → strategy
        self.mapping = {
            ("identity_lookup", "neutral"): "identity_safe",
            ("identity_lookup", "curious"): "identity_safe",
            ("general_query", "confused"): "knowledge_missing",
            ("general_query", "neutral"): "default",
            ("general_query", "urgent"): "urgent_emotion",
            ("general_query", "calm"): "calm_emotion"
        }

    def select(self, features: dict) -> str:
        """
        Selects the response strategy based on features.
        Args:
            features (dict): Dictionary containing
                             'intent', 'emotion', 'urgency', 'follow_up'
        Returns:
            str: Selected system prompt strategy
        """
        intent = features.get("intent", "general_query")
        emotion = features.get("emotion", "neutral")
        urgency = features.get("urgency", "normal")

        # Predictive mapping: intent + emotion
        key = (intent, emotion)
        strategy_key = self.mapping.get(key, "default")

        # Risk-aware override: if info is missing
        if features.get("knowledge_missing", False):
            strategy_key = "knowledge_missing"

        # Urgency-aware override
        if urgency == "high" and strategy_key not in ["identity_safe", "knowledge_missing"]:
            strategy_key = "urgent_emotion"

        return self.strategies.get(strategy_key, self.strategies["default"])


# Example usage
if __name__ == "__main__":
    strategy_selector = ResponseStrategy()

    features_example = {
        "intent": "identity_lookup",
        "emotion": "neutral",
        "urgency": "normal",
        "knowledge_missing": False
    }

    selected_strategy = strategy_selector.select(features_example)
    print("Selected Strategy:", selected_strategy)
