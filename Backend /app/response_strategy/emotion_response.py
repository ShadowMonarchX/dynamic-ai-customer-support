import threading
from typing import Dict, Any

class EmotionResponse:
    def __init__(self):
        self._lock = threading.Lock()

    def get_strategy(self, features: Dict[str, Any]) -> str:
        with self._lock:
            try:
                sentiment = features.get("emotion", "neutral")
                return f"""
                STRATEGY: EMPATHETIC RECOVERY
                The user is feeling {sentiment}. 
                1. Explicitly acknowledge their frustration in the first sentence.
                2. Use an apologetic but professional tone.
                3. Do not use corporate jargon.
                4. Focus on a direct resolution to their problem.
                """
            except Exception as e:
                return "Provide a polite and helpful response."