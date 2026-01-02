import logging
import numpy as np
from typing import Dict, Any
from app.vector_store.faiss_index import FAISSIndex

logger = logging.getLogger(__name__)

class Retriever:
    def __init__(self, faiss_index: FAISSIndex):
        self.index = faiss_index

    def retrieve(self, query_vector: np.ndarray, intent: str, query_text: str = "", max_chunks: int = None) -> Dict[str, Any]:
        try:
            if query_vector is None or len(query_vector) == 0:
                logger.warning("Empty query vector provided")
                return {"docs": [], "count": 0, "status": "empty"}

            query_vector = np.atleast_2d(np.array(query_vector, dtype="float32"))
            if query_vector.shape[1] != self.index.embeddings.shape[1]:
                logger.warning("Query vector dimension mismatch")
                return {"docs": [], "count": 0, "status": "dim_mismatch"}

            return self.index.retrieve(query_vector=query_vector, intent=intent, query_text=query_text, max_chunks=max_chunks or 5)

        except Exception as e:
            logger.error(f"FAISS retrieval failed: {e}")
            return {"docs": [], "count": 0, "status": "error"}
