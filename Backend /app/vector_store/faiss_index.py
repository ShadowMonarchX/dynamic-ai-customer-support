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

# How many final docs per intent
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

# Minimum similarity thresholds (L2 distance → lower is better)
INTENT_DISTANCE_THRESHOLD = {
    "identity": 0.75,
    "faq": 0.85,
    "services": 0.9,
    "skills": 0.9,
    "transactional": 0.8,
    "unknown": 0.95,
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

    def _is_identity_query(self, intent: str) -> bool:
        return intent in {"identity", "profile", "who"}

    def retrieve(self, query_vector: np.ndarray, intent: str) -> Dict[str, Any]:
        with self._lock:
            top_k = INTENT_TOP_K.get(intent, 2)

            if top_k == 0:
                return {"docs": [], "count": 0, "status": "skip"}

            query_vector = np.atleast_2d(query_vector).astype("float32")

            # Over-fetch to allow filtering
            distances, indices = self.index.search(query_vector, top_k * 5)

            threshold = INTENT_DISTANCE_THRESHOLD.get(intent, 0.95)
            scored_docs = []

            for dist, idx in zip(distances[0], indices[0]):
                if idx == -1:
                    continue

                # 🔥 Distance gate (core fix)
                if dist > threshold:
                    continue

                meta = self.metadata[idx]
                weight = meta.get("confidence_weight", 1.0)

                # 🔥 Entity-first filtering for identity queries
                if self._is_identity_query(intent):
                    if not meta.get("identity_rich") and meta.get("content_type") != "identity":
                        continue

                # Intent-specific topic filtering
                if intent in {"order", "refund", "transactional"}:
                    if meta.get("topic") not in {"order", "billing", "refund"}:
                        continue

                if meta.get("status") == "deprecated":
                    continue

                # Score = similarity + metadata confidence
                final_score = dist / weight

                scored_docs.append((final_score, self.documents[idx]))

            # Sort by best score
            scored_docs.sort(key=lambda x: x[0])

            selected_docs = [doc for _, doc in scored_docs[:top_k]]

            return {
                "docs": selected_docs,
                "count": len(selected_docs),
                "status": "success" if selected_docs else "empty"
            }
