import numpy as np
import faiss

class FAISSIndex:
    def __init__(self, embeddings: np.ndarray):
        embeddings = np.atleast_2d(embeddings).astype("float32")
        self.embeddings = embeddings
        self.embedding_dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.index.add(self.embeddings)

    def similarity_search(self, query_vec: np.ndarray, top_k: int = 1):
        query_vec = np.atleast_2d(query_vec).astype("float32")
        return self.index.search(query_vec, top_k)
