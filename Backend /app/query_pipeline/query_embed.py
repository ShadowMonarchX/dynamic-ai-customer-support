import threading
import numpy as np  # type: ignore
import faiss  # type: ignore
from langchain_core.documents import Document  # type: ignore


class QueryEmbedder:
    def __init__(self, embedder):
        self._lock = threading.Lock()
        self.embedder = embedder

    def embed_documents(self, documents):
        """
        Used only if you ever re-embed chunks dynamically.
        """
        with self._lock:
            docs: list[Document] = []

            for d in documents:
                if isinstance(d, Document):
                    docs.append(d)
                else:
                    docs.append(
                        Document(
                            page_content=str(d),
                            metadata=getattr(d, "metadata", {}),
                        )
                    )

            embeddings = self.embedder.embed_documents(docs)

            if embeddings is None or len(embeddings) == 0:
                raise RuntimeError("Document embedding failed")

            vectors = np.atleast_2d(np.array(embeddings, dtype="float32"))
            faiss.normalize_L2(vectors)
            return vectors

    def embed_query(self, query: str) -> np.ndarray:
        """
        Main method used in retrieval pipeline.
        """
        with self._lock:
            if not isinstance(query, str) or not query.strip():
                raise RuntimeError("Empty query")

            embedding = self.embedder.embed_query(query)

            if embedding is None or len(embedding) == 0:
                raise RuntimeError("Query embedding failed")

            vector = np.atleast_2d(np.array(embedding, dtype="float32"))
            faiss.normalize_L2(vector)
            return vector
