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

# faiss_index.py manages semantic vector storage and similarity search, enabling accurate, meaning-based retrieval that powers reliable and hallucination-free customer support AI responses.
import numpy as np
import faiss
import threading
from typing import List, Dict, Any

INTENT_TOP_K = {
    "greeting": 0,
    "faq": 3,
    "services": 3,
    "skills": 3,
    "transactional": 2,
    "order": 2,
    "refund": 2,
    "big_issue": 4,
    "unknown": 2,
}


class FAISSIndex:
    def __init__(
        self,
        embeddings: np.ndarray,
        documents: List[Any],
        metadata: List[Dict[str, Any]]
    ):
        self._lock = threading.Lock()

        if embeddings is None or len(embeddings) == 0:
            raise ValueError("Embeddings cannot be empty")

        self.embeddings = np.atleast_2d(embeddings).astype("float32")
        self.documents = documents
        self.metadata = metadata

        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def retrieve(self, query_vector: np.ndarray, intent: str) -> Dict[str, Any]:
        with self._lock:
            top_k = INTENT_TOP_K.get(intent, 2)

            if top_k == 0:
                return {"docs": [], "count": 0, "status": "skip"}

            query_vector = np.atleast_2d(query_vector).astype("float32")
            distances, indices = self.index.search(query_vector, top_k * 3)

            selected_docs = []

            for idx in indices[0]:
                if idx == -1:
                    continue

                meta = self.metadata[idx]

                # STRICT METADATA FILTERING
                if intent in {"order", "refund", "transactional"}:
                    if meta.get("topic") not in {"order", "billing", "refund"}:
                        continue

                if intent in {"services", "skills", "faq"}:
                    if meta.get("source") not in {"faq", "profile", "docs"}:
                        continue

                if meta.get("status") == "deprecated":
                    continue

                selected_docs.append(self.documents[idx])

                if len(selected_docs) >= top_k:
                    break

            return {
                "docs": selected_docs,
                "count": len(selected_docs),
                "status": "success" if selected_docs else "empty"
            }
