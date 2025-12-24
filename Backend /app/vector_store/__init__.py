"""
vector_store package

Handles vector storage and retrieval using FAISS.
Exports:
- FAISSIndex
"""

from .faiss_index import faiss_index

__all__ = ["FAISSIndex"]
