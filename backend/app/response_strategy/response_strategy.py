from abc import ABC, abstractmethod
from typing import Dict


# -----------------------------
# Base Class
# -----------------------------
class BaseResponseStrategy(ABC):
    """
    Base class for all response strategies.
    Defines HOW LLM should respond, not the answer content.
    """

    @abstractmethod
    def is_applicable(self, features: Dict) -> bool:
        """Check if this strategy applies to the current features"""
        pass

    @abstractmethod
    def system_prompt(self, features: Dict) -> str:
        """Return the system-level instructions for LLM"""
        pass


# -----------------------------
# Concrete Strategies
# -----------------------------
class GreetingStrategy(BaseResponseStrategy):
    def is_applicable(self, features: Dict) -> bool:
        return features.get("is_greeting", False)

    def system_prompt(self, features: Dict) -> str:
        return (
            "You are a professional customer support assistant. "
            "Respond with a brief, friendly greeting (max 15 words) "
            "and ask how you can help. Do not add extra information."
        )


class EmotionStrategy(BaseResponseStrategy):
    def is_applicable(self, features: Dict) -> bool:
        return features.get("emotion") in {"angry", "frustrated"}

    def system_prompt(self, features: Dict) -> str:
        emotion = features.get("emotion", "frustrated")
        return (
            f"The user is feeling {emotion}. "
            "Acknowledge their frustration in the first sentence. "
            "Use a calm, empathetic tone. "
            "Focus on resolving the issue using verified information only."
        )


class FAQStrategy(BaseResponseStrategy):
    def is_applicable(self, features: Dict) -> bool:
        return features.get("intent") in {"faq", "services", "skills", "about", "contact"}

    def system_prompt(self, features: Dict) -> str:
        return (
            "You are a knowledgeable support assistant. "
            "Answer using ONLY the provided context. "
            "Use bullet points or steps if helpful. "
            "If the answer is missing, say you don't have that information. "
            "Keep the response under 150 words."
        )


class TransactionalStrategy(BaseResponseStrategy):
    def is_applicable(self, features: Dict) -> bool:
        return features.get("intent") in {"order", "payment", "refund", "delivery", "transaction"}

    def system_prompt(self, features: Dict) -> str:
        return (
            "You are a professional customer support agent. "
            "Provide a clear, factual response based strictly on retrieved data. "
            "Do not make assumptions or add unnecessary explanations."
        )


class BigIssueStrategy(BaseResponseStrategy):
    def is_applicable(self, features: Dict) -> bool:
        return features.get("urgency") == "high" or features.get("complexity") == "complex"

    def system_prompt(self, features: Dict) -> str:
        return (
            "This is a high-priority or complex issue. "
            "Reassure the user that the issue is being taken seriously. "
            "Explain clearly what can be done now. "
            "If unresolved, guide the user toward escalation or next steps."
        )


class FallbackStrategy(BaseResponseStrategy):
    def is_applicable(self, features: Dict) -> bool:
        return True  # Always applicable as last resort

    def system_prompt(self, features: Dict) -> str:
        return (
            "You are a cautious customer support assistant. "
            "If the information is unclear or missing, "
            "ask a short clarifying question instead of guessing."
        )
