import threading
import numpy as np
from langchain_core.documents import Document
from app.data_ingestion.embedding import Embedder as Embedded
from app.vector_store.faiss_index import FAISSIndex

class RetrievalRouter:
    def __init__(self, embedder: Embedded, vector_store: FAISSIndex):
        self.embedder = embedder
        self.vector_store = vector_store
        self._lock = threading.Lock()

    def retrieve(self, query: str, top_k: int = 5):
        with self._lock:
            try:
                if not query or not query.strip():
                    return []

                # Embed the query
                query_embedding = self.embedder.embed_query(query)
                query_vector = np.atleast_2d(np.array(query_embedding, dtype="float32"))

                import faiss
                faiss.normalize_L2(query_vector)

                # Retrieve from FAISS
                retrieval = self.vector_store.retrieve(
                    query_vector=query_vector,
                    intent="unknown",
                    query_text=query,
                    max_chunks=top_k
                )

                # Convert to list of (Document, similarity)
                results = [(Document(page_content=doc), 1.0) for doc in retrieval.get("docs", [])]
                return results

            except Exception as e:
                raise RuntimeError(f"Retrieval Failed: {e}")
