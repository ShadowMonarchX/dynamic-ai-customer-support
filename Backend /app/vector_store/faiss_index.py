from typing import List
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.embeddings.base import Embeddings


class FAISSIndex:
    def __init__(self, embedding_model: Embeddings):
        self.embedding_model = embedding_model
        self.vectorstore: FAISS | None = None

    def build_index(self, documents: List[Document]) -> None:
        self.vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=self.embedding_model
        )

    def similarity_search(self, query: str, top_k: int = 1) -> List[Document]:
        if not self.vectorstore:
            raise RuntimeError("FAISS index not built")
        return self.vectorstore.similarity_search(query, k=top_k)
