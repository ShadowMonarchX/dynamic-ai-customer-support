# 3. embedding.py
# (Semantic Transformation Layer)
# Purpose
#
# Convert cleaned knowledge into a form the AI can search by meaning.
#
# What This File Represents
#
# Transforms each chunk into a semantic representation
#
# Prepares data for vector-based retrieval
#
# Why This Matters
#
# Keyword search ❌
#
# Meaning-based search ✅
#
# Example:
#
# “Delivery late”
#
# “Order delayed”
# → Treated as the same intent
#
# 📌 This is the core intelligence layer of RAG ingestion


# import threading
# from typing import List, Union
# from langchain_core.documents import Document  # type: ignore
# from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore


# class Embedded:
#     """
#     Intent-aware embedding generator.
#     Preserves identity semantics and reduces weak vector matches.
#     """

#     def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
#         self._lock = threading.Lock()

#         try:
#             self.embedding_model = HuggingFaceEmbeddings(model_name=model_name)
#         except Exception as e:
#             raise RuntimeError(f"Failed to initialize embedding model: {e}")

#     def _apply_embedding_bias(self, text: str, metadata: dict) -> str:
#         """
#         Bias text slightly based on content intent without changing meaning.
#         This improves semantic clustering inside FAISS.
#         """

#         content_type = metadata.get("content_type", "general")
#         identity_rich = metadata.get("identity_rich", False)

#         if content_type == "identity" or identity_rich:
#             # Identity prefix strengthens "Who is X" matching
#             return f"Person Profile: {text}"

#         if content_type in {"support", "technical"}:
#             return f"Support Information: {text}"

#         if content_type in {"how_to", "guide"}:
#             return f"Instructional Content: {text}"

#         return text

#     def _confidence_weight(self, metadata: dict) -> float:
#         """
#         Assign confidence weight per chunk.
#         Used later by FAISS or validator.
#         """
#         if metadata.get("identity_rich"):
#             return 1.2
#         if metadata.get("content_type") == "identity":
#             return 1.15
#         if metadata.get("urgency") == "high":
#             return 1.05
#         return 1.0

#     def embed_documents(self, documents: List[Union[Document, str]]) -> List[List[float]]:
#         with self._lock:
#             try:
#                 docs: List[Document] = [
#                     doc if isinstance(doc, Document)
#                     else Document(page_content=str(doc), metadata={})
#                     for doc in documents
#                 ]

#                 biased_texts = []
#                 for doc in docs:
#                     biased_text = self._apply_embedding_bias(
#                         doc.page_content,
#                         doc.metadata or {}
#                     )

#                     # Store weight for downstream usage
#                     doc.metadata["confidence_weight"] = self._confidence_weight(
#                         doc.metadata or {}
#                     )

#                     biased_texts.append(biased_text)

#                 return self.embedding_model.embed_documents(biased_texts)

#             except Exception as e:
#                 raise RuntimeError(f"Document embedding failed: {e}")

#     def embed_query(self, query: str) -> List[float]:
#         with self._lock:
#             try:
#                 if not query.strip():
#                     raise ValueError("Query string is empty")

#                 # Normalize query intent slightly (NO hallucination)
#                 normalized_query = query.strip()

#                 if normalized_query.lower().startswith("who is"):
#                     normalized_query = f"Person Profile Query: {normalized_query}"

#                 return self.embedding_model.embed_query(normalized_query)

#             except Exception as e:
#                 raise RuntimeError(f"Query embedding failed: {e}")

import threading
from typing import List, Union
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np
import faiss


class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self._lock = threading.Lock()
        self.model = HuggingFaceEmbeddings(model_name=model_name)

    def _apply_bias(self, text: str, meta: dict) -> str:
        if meta.get("identity_rich") or meta.get("content_type") == "identity":
            return f"Person Profile: {text}"
        if meta.get("category") == "technical":
            return f"Technical Support Information: {text}"
        if meta.get("category") == "billing":
            return f"Billing Information: {text}"
        return text

    def embed_documents(self, documents: List[Union[Document, str]]) -> np.ndarray:
        with self._lock:
            if not documents:
                raise RuntimeError("No documents to embed")

            docs: List[Document] = []
            for d in documents:
                if isinstance(d, Document):
                    docs.append(d)
                else:
                    docs.append(Document(page_content=str(d), metadata={}))

            texts = []
            for doc in docs:
                meta = dict(doc.metadata or {})
                text = self._apply_bias(doc.page_content, meta)
                if not text.strip():
                    text = "empty"
                texts.append(text)
                doc.metadata = meta

            embeddings = self.model.embed_documents(texts)
            embeddings = np.asarray(embeddings, dtype=np.float32)

            if embeddings.ndim != 2:
                raise RuntimeError("Invalid embedding shape")

            faiss.normalize_L2(embeddings)
            return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        with self._lock:
            q = query.strip()
            if not q:
                raise RuntimeError("Empty query")

            if q.lower().startswith("who is"):
                q = f"Person Profile Query: {q}"

            vec = self.model.embed_query(q)
            vec = np.asarray(vec, dtype=np.float32).reshape(1, -1)
            faiss.normalize_L2(vec)
            return vec


from .data_load import DataSource
from .preprocessing import Preprocessor

DATA_PATH = "/Users/jenishshekhada/Desktop/Inten/dynamic-ai-customer-support/backend /data/new_training_data.txt"


data_source = DataSource(DATA_PATH)
documents = data_source.load()
print(f"\nLoaded {len(documents)} raw documents\n")

# 2. Preprocess (CLEAN + CHUNK)
preprocessor = Preprocessor()
processed_docs = preprocessor.transform_documents(documents)

print(f"Processed into {len(processed_docs)} cleaned chunks\n")

# ✅ PRINT ALL CLEAN TEXT (YOU ASKED THIS)
print("\n========== CLEANED TEXT OUTPUT ==========\n")

for idx, doc in enumerate(processed_docs, start=1):
    print(f"----- Chunk {idx} -----")
    print(doc.page_content)
    print(f"Metadata: {doc.metadata}")
    print()


# for doc in processed_documents:
#     print(doc)
