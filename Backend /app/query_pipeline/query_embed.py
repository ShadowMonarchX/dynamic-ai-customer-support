# ## Step 3: Intelligent Retrieval from Your Data

# ### Folder

# `query_pipeline/`

# ### Files Involved

# * `query_embed.py`
# * `retrieval_router.py`

# ### What Happens Here

# The bot searches **only your internal data**, such as:

# * Website FAQs
# * Help center articles
# * Order and delivery policies
# * Past support tickets
# * Product documentation
# * App database summaries

# The system:

# * Retrieves **multiple relevant documents**
# * Ranks them by **semantic relevance**
# * Filters **outdated or irrelevant** information

# 📌 This step is critical for **hallucination prevention**.


import threading
from typing import List


class QueryEmbedder:
    def __init__(self, model):
        self.model = model
        self._lock = threading.Lock()

    def embed(self, query_text: str) -> List[float]:
        with self._lock:
            query = self._normalize(query_text)

            if not query:
                raise ValueError("Empty query after normalization")

            return self.model.embed_query(query)

    def _normalize(self, text: str) -> str:
        text = text.strip().lower()

        # Remove noise
        for ch in ["?", "!", ".", ",", ";"]:
            text = text.replace(ch, "")

        # Normalize common user phrasing
        replacements = {
            "who is": "",
            "tell me about": "",
            "can you explain": "",
        }

        for k, v in replacements.items():
            text = text.replace(k, v)

        return text.strip()
