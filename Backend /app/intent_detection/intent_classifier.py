from typing import Dict, Any
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_community.llms import Ollama

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
    lowered = text.lower().strip()
    word_count = len(lowered.split())

    if word_count <= 2 and any(g in lowered for g in ["hi", "hey", "hello"]):
        return {"intent": "greeting", "emotion": "neutral", "urgency": "low", "complexity": "small"}

    if word_count < 12 and any(w in lowered for w in ["what is", "how does", "policy", "price", "cost"]):
        return {"intent": "faq", "emotion": "neutral", "urgency": "low", "complexity": "small"}

    if any(w in lowered for w in ["refund", "cancel", "return", "order", "track"]):
        return {"intent": "transactional", "emotion": "neutral", "urgency": "medium", "complexity": "medium"}

    if any(w in lowered for w in ["angry", "ridiculous", "worst", "third time", "not working", "failed"]):
        return {"intent": "account_support", "emotion": "frustrated", "urgency": "high", "complexity": "medium"}

    return {}

class IntentClassifier:
    def __init__(self, model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
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
        if not isinstance(message, str) or not message.strip():
            return {"intent": "unknown", "emotion": "neutral", "urgency": "low", "complexity": "small"}

        rule_result = _basic_rules(message)
        if rule_result:
            return rule_result

        chain = self.prompt | self.llm | self.parser
        try:
            result = chain.invoke({"message": message})
        except OutputParserException:
            result = {}

        intent = result.get("intent", "unknown")
        emotion = result.get("emotion", "neutral")
        urgency = result.get("urgency", "low")
        complexity = result.get("complexity", "small")

        if intent not in INTENTS:
            intent = "unknown"
        if emotion not in EMOTIONS:
            emotion = "neutral"
        if urgency not in {"low", "medium", "high"}:
            urgency = "low"
        if complexity not in {"small", "medium", "big"}:
            complexity = "small"

        return {"intent": intent, "emotion": emotion, "urgency": urgency, "complexity": complexity}
