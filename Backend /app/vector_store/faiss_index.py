import numpy as np
import faiss
from typing import List, Dict, Any

INTENT_TOP_K = {
    "greeting": 0,
    "identity": 2,
    "faq": 3,
    "services": 3,
    "skills": 3,
    "transactional": 2,
    "order": 2,
    "refund": 2,
    "billing/refund": 2,
    "order/delivery": 2,
    "general": 2,
    "unknown": 2,
}

INTENT_SIMILARITY_THRESHOLD = {
    "identity": 0.35,
    "faq": 0.55,
    "services": 0.55,
    "skills": 0.55,
    "transactional": 0.5,
    "billing/refund": 0.5,
    "order/delivery": 0.5,
    "general": 0.45,
    "unknown": 0.45,
}

INTENT_TOPIC_MAP = {
    "order": {"order", "delivery"},
    "refund": {"billing", "refund"},
    "billing/refund": {"billing", "refund"},
    "transactional": {"order", "billing", "refund"},
}


class FAISSIndex:
    def __init__(
        self,
        embeddings: np.ndarray,
        documents: List[Any],
        metadata: List[Dict[str, Any]],
        hnsw_m: int = 32,
        ef_search: int = 64,
        ef_construction: int = 200,
    ):
        if embeddings is None or len(embeddings) == 0:
            raise ValueError("Embeddings cannot be empty")
        if len(embeddings) != len(documents) or len(embeddings) != len(metadata):
            raise ValueError("Embeddings, documents, and metadata length mismatch")
        embeddings = np.atleast_2d(embeddings).astype("float32")
        faiss.normalize_L2(embeddings)
        self.embeddings = embeddings
        self.documents = documents
        self.metadata = metadata
        dim = embeddings.shape[1]
        self.index = faiss.IndexHNSWFlat(dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efSearch = ef_search
        self.index.hnsw.efConstruction = ef_construction
        self.index.add(self.embeddings)

    def _is_identity_query(self, intent: str, query_text: str) -> bool:
        return intent == "identity" or query_text.lower().startswith("who is")

    def retrieve(
        self,
        query_vector: np.ndarray,
        intent: str,
        query_text: str = "",
        top_k: int = 2,
    ) -> Dict[str, Any]:
        top_k = top_k or INTENT_TOP_K.get(intent, 2)
        if top_k == 0:
            return {"docs": [], "count": 0, "status": "skip"}
        if query_vector is None or len(query_vector) == 0:
            raise ValueError("Query embedding is empty")
        query_vector = np.atleast_2d(query_vector).astype("float32")
        if query_vector.shape[1] != self.index.d:
            raise ValueError(
                f"Query vector dim {query_vector.shape[1]} ≠ index dim {self.index.d}"
            )
        faiss.normalize_L2(query_vector)
        distances, indices = self.index.search(query_vector, top_k * 10)
        threshold = INTENT_SIMILARITY_THRESHOLD.get(intent, 0.45)
        scored_docs = []
        for sim, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.documents):
                continue
            meta = self.metadata[idx] or {}
            if meta.get("status") == "deprecated":
                continue
            if self._is_identity_query(intent, query_text):
                if "identity" not in meta.get("content_type", "general"):
                    continue
            allowed_topics = INTENT_TOPIC_MAP.get(intent)
            if allowed_topics and meta.get("topic", "general") not in allowed_topics:
                continue
            if sim < threshold:
                continue
            weight = float(meta.get("confidence_weight", 1.0))
            weight = max(0.1, min(weight, 2.0))
            scored_docs.append((sim * weight, self.documents[idx]))
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        selected_docs = [doc for _, doc in scored_docs[:top_k]]
        if not selected_docs and len(distances[0]) > 0:
            for sim, idx in zip(distances[0], indices[0]):
                if idx >= 0 and idx < len(self.documents):
                    selected_docs.append(self.documents[idx])
                    if len(selected_docs) >= top_k:
                        break
        return {
            "docs": selected_docs,
            "count": len(selected_docs),
            "status": "success" if selected_docs else "empty",
        }
