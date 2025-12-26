from typing import List
from langchain_community.vectorstores import FAISS  # type: ignore
from langchain_core.documents import Document  # type: ignore
from langchain_community.embeddings import HuggingFaceEmbeddings  # correct import


class FAISSIndex:
    def __init__(self, embedding_model: HuggingFaceEmbeddings):
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
