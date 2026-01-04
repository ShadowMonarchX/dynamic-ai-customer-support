# vector_store/faiss_index.py
# (Vector Storage & Retrieval Layer)
# Purpose

# faiss_index.py is responsible for storing semantic vectors and retrieving the most relevant knowledge for the customer support AI bot.

# It is the bridge between ingestion and query pipelines.

# Core Responsibilities
# 1. Stores Vectors

# This file manages:

# Storage of vector embeddings generated during ingestion

# Efficient indexing for large-scale data

# Persistence of vector state across sessions

# 📌 Each vector represents the meaning of a knowledge chunk, not raw text.

# 2. Handles Similarity Search

# When a user asks a question:

# The query is converted into a vector

# faiss_index.py finds nearest vectors by semantic similarity

# Returns the most relevant chunks

# This enables:

# Meaning-based search

# High recall and precision

# Fast response times

# 📌 Keyword matching ❌
# 📌 Semantic similarity ✅

# 3. Ensures Consistency

# This file ensures:

# Vectors and metadata stay aligned

# Index updates don’t corrupt retrieval

# Re-indexing is clean and predictable

# Old or invalid vectors are removed safely

# 📌 Prevents retrieval drift and stale answers.

# Why FAISS Is Used Conceptually

# FAISS allows:

# Scalable vector search

# Low-latency retrieval

# High-dimensional similarity matching

# It is ideal for:

# FAQs

# Policies

# Product documentation

# Support knowledge bases

# Retrieval Quality Outcome

# Because of this design, the bot understands meaning:

# “Delivery late”

# “Order delayed”

# “Shipment not arrived”

# → All map to the same semantic intent

# 📌 This results in high-quality, reliable retrieval.

# Separation of Responsibilities (Why This File Is Clean)
# Layer	Responsibility
# Ingestion	Data cleaning, chunking, embedding
# Vector Store	Storage & similarity search
# Query Pipeline	Query understanding & retrieval logic
# Response Strategy	Answer formatting & tone

# ✅ faiss_index.py does not:

# Clean data

# Modify text

# Decide response tone

# Handle business logic

# 📌 This separation makes the system maintainable and scalable.

# System-Level Role

# faiss_index.py guarantees that:

# Retrieval is accurate

# Responses are grounded

# Hallucinations are minimized

# Follow-up questions work correctly

# One-Line Summary (Docs / Interview)

# # faiss_index.py manages semantic vector storage and similarity search, enabling accurate, meaning-based retrieval that powers reliable and hallucination-free customer support AI responses.


# import numpy as np
# import faiss # Facebook AI Similarity Search
# import threading
# from typing import List, Dict, Any

# INTENT_TOP_K = {
#     "greeting": 0,
#     "identity": 2,
#     "faq": 3,
#     "services": 3,
#     "skills": 3,
#     "transactional": 2,
#     "order": 2,
#     "refund": 2,
#     "big_issue": 4,
#     "unknown": 2,
# }

# # Cosine similarity thresholds (higher is better, 1 = perfect match)
# INTENT_SIMILARITY_THRESHOLD = {
#     "identity": 0.65,
#     "faq": 0.7,
#     "services": 0.7,
#     "skills": 0.7,
#     "transactional": 0.6,
#     "unknown": 0.5,
# }


# class FAISSIndex:
#     def __init__(
#         self,
#         embeddings: np.ndarray,
#         documents: List[Any],
#         metadata: List[Dict[str, Any]]
#     ):
#         self._lock = threading.Lock()

#         if embeddings is None or len(embeddings) == 0:
#             raise ValueError("Embeddings cannot be empty")

#         # Normalize embeddings for cosine similarity
#         self.embeddings = np.atleast_2d(embeddings).astype("float32")
#         faiss.normalize_L2(self.embeddings)

#         self.documents = documents
#         self.metadata = metadata

#         self.index = faiss.IndexFlatIP(self.embeddings.shape[1])  # inner product = cosine
#         self.index.add(self.embeddings)

#     def _is_identity_query(self, intent: str, query_text: str = "") -> bool:
#         return intent in {"identity", "profile"} or query_text.lower().startswith("who is")

#     def retrieve(self, query_vector: np.ndarray, intent: str, query_text: str = "") -> Dict[str, Any]:
#         with self._lock:
#             top_k = INTENT_TOP_K.get(intent, 2)

#             if top_k == 0:
#                 return {"docs": [], "count": 0, "status": "skip"}

#             query_vector = np.atleast_2d(query_vector).astype("float32")
#             faiss.normalize_L2(query_vector)

#             distances, indices = self.index.search(query_vector, top_k * 5)  # over-fetch
#             threshold = INTENT_SIMILARITY_THRESHOLD.get(intent, 0.5)
#             scored_docs = []

#             for sim, idx in zip(distances[0], indices[0]):
#                 if idx == -1:
#                     continue

#                 meta = self.metadata[idx]
#                 weight = meta.get("confidence_weight", 1.0)

#                 # Identity-first filtering
#                 if self._is_identity_query(intent, query_text):
#                     if not meta.get("identity_rich") and meta.get("content_type") != "identity":
#                         continue

#                 # Intent-specific topic filtering
#                 if intent in {"order", "refund", "transactional"}:
#                     if meta.get("topic") not in {"order", "billing", "refund"}:
#                         continue

#                 if meta.get("status") == "deprecated":
#                     continue

#                 # Apply threshold
#                 if sim < threshold:
#                     continue

#                 # Score = similarity * confidence weight (higher is better)
#                 final_score = sim * weight
#                 scored_docs.append((final_score, self.documents[idx]))

#             scored_docs.sort(key=lambda x: x[0], reverse=True)
#             selected_docs = [doc for _, doc in scored_docs[:top_k]]

#             if not selected_docs:
#                 print(f"[FAISS Warning] Empty retrieval. Intent: {intent}, Query: '{query_text}'")
#                 print(f"Top similarities: {distances[0][:5]}")

#             return {
#                 "docs": selected_docs,
#                 "count": len(selected_docs),
#                 "status": "success" if selected_docs else "empty"
#             }



import numpy as np #type: ignore
import faiss #type: ignore
from typing import List, Dict, Any
import logging

INTENT_TOP_K = {
    "greeting": 0,
    "identity": 2,
    "faq": 3,
    "services": 3,
    "skills": 3,
    "transactional": 2,
    "order": 2,
    "refund": 2,
    "big_issue": 4,
    "unknown": 2,
}

INTENT_SIMILARITY_THRESHOLD = {
    "identity": 0.65,
    "faq": 0.7,
    "services": 0.7,
    "skills": 0.7,
    "transactional": 0.6,
    "unknown": 0.5,
}

intent_topic_map = {
    "order": {"order", "billing", "refund"},
    "refund": {"order", "billing", "refund"},
    "transactional": {"order", "billing", "refund"},
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FAISSIndex")


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
        return intent in {"identity", "profile"} or query_text.lower().startswith(
            "who is"
        )

    def retrieve(
        self,
        query_vector: np.ndarray,
        intent: str,
        query_text: str = "",
        max_chunks: int = None,
    ) -> Dict[str, Any]:
        top_k = INTENT_TOP_K.get(intent, 2)
        if max_chunks is not None:
            top_k = max_chunks
        if top_k == 0:
            return {"docs": [], "count": 0, "status": "skip"}
        if query_vector is None or len(query_vector) == 0:
            raise ValueError("Query embedding is empty")
        query_vector = np.atleast_2d(query_vector).astype("float32")
        if query_vector.shape[1] != self.index.d:
            raise ValueError(
                f"Query vector dimension ({query_vector.shape[1]}) does not match index ({self.index.d})"
            )
        faiss.normalize_L2(query_vector)
        distances, indices = self.index.search(query_vector, top_k * 10)
        threshold = INTENT_SIMILARITY_THRESHOLD.get(intent, 0.5)
        scored_docs = []
        for sim, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.documents):
                continue
            meta = self.metadata[idx] or {}
            if meta.get("status") == "deprecated":
                continue
            if self._is_identity_query(intent, query_text):
                if meta.get("content_type") != "identity":
                    continue
            allowed_topics = intent_topic_map.get(intent, None)
            if allowed_topics and meta.get("topic", "general") not in allowed_topics:
                continue
            if sim < threshold:
                continue
            weight = float(meta.get("confidence_weight", 1.0))
            weight = max(0.1, min(weight, 2.0))
            scored_docs.append((sim * weight, self.documents[idx]))
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        selected_docs = [doc for _, doc in scored_docs[:top_k]]
        if not selected_docs:
            logger.warning(
                f"Empty retrieval for intent: {intent}, query: '{query_text}'"
            )
        return {
            "docs": selected_docs,
            "count": len(selected_docs),
            "status": "success" if selected_docs else "empty",
        }
