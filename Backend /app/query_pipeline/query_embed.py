# retrieval_router.py
# (Intelligent Retrieval Layer)
# Purpose
#
# Search and retrieve the most relevant internal documents based on query embeddings.
#
# What Happens Here
#
# Retrieves multiple relevant documents
# Ranks them by semantic relevance
# Filters outdated or irrelevant information
#
# This step is critical for hallucination prevention.

import threading
from typing import List, Tuple
from langchain_core.documents import Document  # type: ignore
from query_embed import QueryEmbedder  # type: ignore

class RetrievalRouter:
    def __init__(self, embedder: QueryEmbedder, vector_store):
        self.embedder = embedder
        self.vector_store = vector_store
        self._lock = threading.Lock()

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        """
        Retrieve top_k most relevant documents for the given query.
        Returns a list of tuples: (Document, similarity_score)
        """
        with self._lock:
            try:
                if not query or not query.strip():
                    return []

                # Embed the query
                query_embedding = self.embedder.embed(query)

                # Perform similarity search in the vector store
                results = self.vector_store.similarity_search_by_vector(
                    vector=query_embedding,
                    k=top_k
                )

                return results

            except Exception as e:
                raise RuntimeError(f"Retrieval Failed: {e}")
