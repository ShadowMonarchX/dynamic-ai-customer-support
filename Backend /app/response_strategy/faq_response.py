
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