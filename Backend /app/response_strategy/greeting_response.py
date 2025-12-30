import threading

class GreetingResponse:
    def __init__(self):
        self._lock = threading.Lock()

    def get_strategy(self, features: dict) -> str:
        with self._lock:
            return (
                "STRATEGY: GREETING. You are a professional AI support bot. "
                "Be extremely brief (max 2 sentences). Greeting the user and "
                "ask how you can help with their technical or account issues. "
                "DO NOT write a long letter or personal stories."
            )