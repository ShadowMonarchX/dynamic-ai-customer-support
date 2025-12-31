


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