import numpy as np # type: ignore
import faiss # type: ignore

class FAISSIndex:
    def __init__(self, embeddings):
        self.embeddings = np.asarray(embeddings, dtype="float32")
        self.embedding_dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.index.add(self.embeddings)

    def search(self, query_vec, top_k=1):
        query = np.asarray(query_vec, dtype="float32").reshape(1, -1)
        return self.index.search(query, top_k)
