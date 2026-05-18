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

