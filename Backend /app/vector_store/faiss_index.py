# import numpy as np
# import faiss
# from typing import List, Dict, Any

# INTENT_TOP_K = {
#     "greeting": 0,
#     "faq": 1,
#     "transactional": 0,
#     "big_issue": 4,
#     "account_support": 2,
#     "emotion": 1,
#     "unknown": 1,
# }

# class FAISSIndex:
#     def __init__(self, embeddings: np.ndarray, documents: List[str], metadata: List[Dict[str, Any]]):
#         if embeddings is None or len(embeddings) == 0:
#             raise ValueError("Embeddings cannot be empty")
#         self.embeddings = np.atleast_2d(embeddings).astype("float32")
#         self.documents = documents
#         self.metadata = metadata
#         self.embedding_dim = self.embeddings.shape[1]
#         self.index = faiss.IndexFlatL2(self.embedding_dim)
#         self.index.add(self.embeddings)

#     def retrieve(self, query_vector: np.ndarray, intent: str) -> Dict[str, Any]:
#         top_k = INTENT_TOP_K.get(intent, 1)
#         if top_k == 0:
#             return {"context": "", "chunks_used": 0, "reason": "Retrieval disabled for this intent"}
#         query_vector = np.atleast_2d(query_vector).astype("float32")
#         distances, indices = self.index.search(query_vector, top_k)
#         retrieved_chunks = []
#         for idx in indices[0]:
#             if 0 <= idx < len(self.documents) and self.metadata[idx]["intent"] == intent:
#                 retrieved_chunks.append(self.documents[idx])
#         context = "\n\n".join(retrieved_chunks)
#         return {"context": context, "chunks_used": len(retrieved_chunks), "reason": f"Retrieved {len(retrieved_chunks)} chunks for intent '{intent}'"}

# class QueryEmbedder:
#     def __init__(self, model):
#         self.model = model

#     def embed(self, query_text: str):
#         return self.model.embed_query(query_text)


import numpy as np
import faiss
import threading
from typing import List, Dict, Any

INTENT_TOP_K = {
    "greeting": 0,
    "faq": 3,
    "transactional": 2,
    "big_issue": 5,
    "account_support": 3,
    "unknown": 2,
}

class FAISSIndex:
    def __init__(self, embeddings: np.ndarray, documents: List[Any], metadata: List[Dict[str, Any]]):
        self._lock = threading.Lock()
        try:
            if embeddings is None or len(embeddings) == 0:
                raise ValueError("Embeddings cannot be empty")
            
            self.embeddings = np.atleast_2d(embeddings).astype("float32")
            self.documents = documents
            self.metadata = metadata
            self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
            self.index.add(self.embeddings)
        except Exception as e:
            raise RuntimeError(f"FAISS Initialization Failed: {e}")

    def retrieve(self, query_vector: np.ndarray, intent: str) -> Dict[str, Any]:
        with self._lock:
            try:
                top_k = INTENT_TOP_K.get(intent, 2)
                if top_k == 0:
                    return {"docs": [], "count": 0, "status": "no_retrieval_needed"}

                query_vector = np.atleast_2d(query_vector).astype("float32")
                distances, indices = self.index.search(query_vector, top_k * 2) 

                retrieved_docs = []
                for idx in indices[0]:
                    if idx == -1: continue
                    
                    doc_meta = self.metadata[idx]
                    # Metadata Filtering Logic
                    if intent == "transactional" and doc_meta.get("category") != "billing":
                        continue
                    
                    retrieved_docs.append(self.documents[idx])
                    if len(retrieved_docs) >= top_k:
                        break

                return {
                    "docs": retrieved_docs,
                    "count": len(retrieved_docs),
                    "status": "success"
                }
            except Exception as e:
                raise RuntimeError(f"Retrieval Error: {e}")