import numpy as np
import faiss

class FAISSIndex:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.embedding_dim = embeddings[0].shape[0]
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.index.add(np.array(embeddings).astype("float32"))

    def search(self, query_vec, top_k=3):
        query_vec = query_vec.astype("float32").reshape(1, -1)
        D, I = self.index.search(query_vec, top_k)
        return D, I
