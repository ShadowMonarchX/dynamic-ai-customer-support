import numpy as np  # type: ignore
import faiss # type: ignore


class FAISSIndex:
    def __init__(self, embeddings: np.ndarray):
        embeddings = np.asarray(embeddings, dtype="float32")

        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array")

        self.embeddings = embeddings
        self.embedding_dim = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.index.add(self.embeddings)

    def similarity_search(self, query_vec: np.ndarray, top_k: int = 1):
        query_vec = np.asarray(query_vec, dtype="float32")

        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)

        if query_vec.shape[1] != self.embedding_dim:
            raise ValueError("Query vector dimension mismatch")

        return self.index.search(query_vec, top_k)
