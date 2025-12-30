import threading
from typing import Dict, Any

class FAQResponse:
    def __init__(self):
        self._lock = threading.Lock()

    def get_strategy(self, features: Dict[str, Any]) -> str:
        with self._lock:
            return """
            STRATEGY: KNOWLEDGE RETRIEVAL
            1. Use the provided context to answer the question accurately.
            2. Present steps in a numbered or bulleted list for readability.
            3. If the answer is not in the context, state that you don't have that information.
            4. Keep the response under 150 words.
            """