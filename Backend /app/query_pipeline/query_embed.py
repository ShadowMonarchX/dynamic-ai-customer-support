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
from typing import List, Union
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np
import faiss

class Embedded:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self._lock = threading.Lock()
        self.embedding_model = HuggingFaceEmbeddings(model_name=model_name)

    def _apply_embedding_bias(self, text: str, metadata: dict) -> str:
        content_type = metadata.get("content_type", "general")
        identity_rich = metadata.get("identity_rich", False)
        if content_type == "identity" or identity_rich:
            return f"Person Profile: {text}"
        if content_type in {"support", "technical"}:
            return f"Support Information: {text}"
        if content_type in {"how_to", "guide"}:
            return f"Instructional Content: {text}"
        return text

    def _confidence_weight(self, metadata: dict) -> float:
        if metadata.get("identity_rich"):
            return 1.2
        if metadata.get("content_type") == "identity":
            return 1.15
        if metadata.get("urgency") == "high":
            return 1.05
        return 1.0

    def embed_documents(self, documents: List[Union[Document, str]]) -> np.ndarray:
        with self._lock:
            docs = [
                doc if isinstance(doc, Document)
                else Document(page_content=str(doc), metadata={})
                for doc in documents
            ]
            texts = []
            for doc in docs:
                metadata = dict(doc.metadata or {})
                text = self._apply_embedding_bias(doc.page_content, metadata)
                metadata["confidence_weight"] = self._confidence_weight(metadata)
                doc.metadata = metadata
                texts.append(text)
            embeddings = self.embedding_model.embed_documents(texts)
            embeddings = np.atleast_2d(np.array(embeddings, dtype=np.float32))
            faiss.normalize_L2(embeddings)
            return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        with self._lock:
            q = query.strip()
            if not q:
                raise RuntimeError("Empty query")
            if q.lower().startswith("who is"):
                q = f"Person Profile Query: {q}"
            embedding = self.embedding_model.embed_query(q)
            embedding = np.atleast_2d(np.array(embedding, dtype=np.float32))
            faiss.normalize_L2(embedding)
            return embedding
