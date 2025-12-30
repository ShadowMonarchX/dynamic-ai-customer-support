import threading

class BigIssueResponse:
    def __init__(self):
        self._lock = threading.Lock()

    def get_strategy(self, features: dict) -> str:
        with self._lock:
            return """
            STRATEGY: CRISIS ESCALATION
            1. Reassure the user that their issue is being prioritized.
            2. Explain that this is a complex case.
            3. Provide a temporary workaround if available in the context.
            4. Inform them that a human specialist may need to review this.
            """