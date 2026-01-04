import threading
import numpy as np #type: ignore
from langchain_core.documents import Document #type: ignore


class QueryEmbedder:
    def __init__(self, embedder):
        self._lock = threading.Lock()
        self.embedder = embedder

    def embed_documents(self, documents):
        with self._lock:
            docs = [
                d if isinstance(d, Document)
                else Document(page_content=str(d), metadata={})
                for d in documents
            ]
            embeddings = self.embedder.embed_documents(docs)
            return embeddings

    def embed_query(self, query: str):
        with self._lock:
            if not query or not query.strip():
                raise RuntimeError("Empty query")
            embedding = self.embedder.embed_query(query)
            return np.atleast_2d(np.array(embedding, dtype="float32"))
