import threading
from typing import Dict, Any
from app.query_pipeline.query_embed import QueryEmbedder
from app.vector_store.faiss_index import FAISSIndex
import numpy as np


class RetrievalRouter:
    def __init__(self, query_embedder: QueryEmbedder, vector_store: FAISSIndex):
        self.query_embedder = query_embedder
        self.vector_store = vector_store
        self._lock = threading.Lock()

    def retrieve(
        self, query: str, intent: str = "unknown", top_k: int = 5
    ) -> Dict[str, Any]:
        with self._lock:
            if not query or not query.strip():
                return {"docs": [], "count": 0, "status": "empty"}

            query_vector = self.query_embedder.embed_query(query)
            query_vector = np.atleast_2d(query_vector).astype("float32")

            retrieval = self.vector_store.retrieve(
                query_vector=query_vector,
                intent=intent,
                query_text=query,
                top_k=top_k,
            )

            if "docs" not in retrieval:
                retrieval["docs"] = []
            if "count" not in retrieval:
                retrieval["count"] = len(retrieval.get("docs", []))
            if "status" not in retrieval:
                retrieval["status"] = "success" if retrieval["docs"] else "empty"

            return retrieval
