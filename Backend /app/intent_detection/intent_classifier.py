from typing import Dict, Any
from langchain_core.prompts import PromptTemplate  # type: ignore
from langchain_core.output_parsers import JsonOutputParser  # type: ignore
from langchain_core.exceptions import OutputParserException  # type: ignore
from langchain_community.llms import Ollama  # type: ignore

INTENTS = {
    "greeting",
    "faq",
    "transactional",
    "big_issue",
    "account_support",
    "unknown"
}

EMOTIONS = {
    "neutral",
    "frustrated",
    "angry",
    "urgent",
    "stressed"
}


def _basic_rules(text: str) -> Dict[str, Any]:
    """
    Quick local rules for common query types
    """
    lowered = text.lower().strip()
    word_count = len(lowered.split())

    # Greeting
    if word_count <= 2 and any(g in lowered for g in ["hi", "hey", "hello"]):
        return {"intent": "greeting", "emotion": "neutral", "urgency": "low", "complexity": "small"}

    # FAQ
    if word_count < 12 and any(w in lowered for w in ["what is", "how does", "policy", "price", "cost"]):
        return {"intent": "faq", "emotion": "neutral", "urgency": "low", "complexity": "small"}

    # Transactional
    if any(w in lowered for w in ["refund", "cancel", "return", "order", "track"]):
        return {"intent": "transactional", "emotion": "neutral", "urgency": "medium", "complexity": "medium"}

    # Account Support / Frustration
    if any(w in lowered for w in ["angry", "ridiculous", "worst", "third time", "not working", "failed"]):
        return {"intent": "account_support", "emotion": "frustrated", "urgency": "high", "complexity": "medium"}

    return {}


class IntentClassifier:
    """
    Returns intent, emotion, urgency, complexity for a user query
    """

    def __init__(self, model: str = "llama3"):
        self.llm = Ollama(model=model, temperature=0)
        self.parser = JsonOutputParser()
        self.prompt = PromptTemplate(
            template="""
                Return ONLY valid JSON.
                No markdown. No explanations.

                Fields:
                intent: one of {intents}
                emotion: one of {emotions}
                urgency: low | medium | high
                complexity: small | medium | big

                User message:
                "{message}"
            """,
            input_variables=["message"],
            partial_variables={
                "intents": ", ".join(INTENTS),
                "emotions": ", ".join(EMOTIONS)
            }
        )

    def classify(self, message: str) -> Dict[str, Any]:
        # Empty or invalid message
        if not isinstance(message, str) or not message.strip():
            return {"intent": "unknown", "emotion": "neutral", "urgency": "low", "complexity": "small"}

        # First, try simple rules
        rule_result = _basic_rules(message)
        if rule_result:
            return rule_result

        # Use LLM fallback
        chain = self.prompt | self.llm | self.parser
        try:
            result = chain.invoke({"message": message})
        except OutputParserException:
            result = {}

        # Ensure all keys exist with defaults
        intent = result.get("intent", "unknown")
        emotion = result.get("emotion", "neutral")
        urgency = result.get("urgency", "low")
        complexity = result.get("complexity", "small")

        # Validate values
        if intent not in INTENTS:
            intent = "unknown"
        if emotion not in EMOTIONS:
            emotion = "neutral"
        if urgency not in {"low", "medium", "high"}:
            urgency = "low"
        if complexity not in {"small", "medium", "big"}:
            complexity = "small"

        return {"intent": intent, "emotion": emotion, "urgency": urgency, "complexity": complexity}
