
# ## Step 5: Answer Generation (Grounded)

# ### Folder

# `response_strategy/`

# ### Files Involved

# * `faq_response.py`
# * `transactional_response.py`
# * `emotion_response.py`

# ### What Happens Here

# The AI generates the final answer:

# * Uses **only retrieved and validated content**
# * Matches **customer support tone**
# * Keeps answers **short, clear, and helpful**
# * Avoids technical or internal jargon

# ### Example Output

# > “Your order is delayed due to high demand. Our standard delivery time is 5–7 business days. You’ll receive a tracking update soon.”


import threading
from typing import Dict, Any

class TransactionalResponse:
    def __init__(self):
        self._lock = threading.Lock()

    def get_strategy(self, features: Dict[str, Any]) -> str:
        with self._lock:
            urgency = features.get("urgency", "medium")
            return f"""
            STRATEGY: TRANSACTIONAL SUPPORT (Urgency: {urgency})
            1. Focus strictly on policy details and data.
            2. Clearly state timelines (e.g., '3-5 business days').
            3. If an Order ID is missing, ask for it politely.
            4. Ensure the tone is efficient and precise.
            """