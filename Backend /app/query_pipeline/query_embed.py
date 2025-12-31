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
            try:
                if not query_text or not query_text.strip():
                    raise ValueError("Query text cannot be empty for embedding")
                
                return self.model.embed_query(query_text)
                
            except ValueError as ve:
                raise RuntimeError(f"Validation Error: {ve}")
            except Exception as e:
                raise RuntimeError(f"Model Embedding Failed: {e}")