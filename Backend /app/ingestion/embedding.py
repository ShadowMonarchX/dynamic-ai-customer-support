# from typing import List
# from langchain_core.documents import Document # type: ignore
# from langchain_huggingface import HuggingFaceEmbeddings # type: ignore

# class Embedded:
#     def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
#         self.embedding_model = HuggingFaceEmbeddings(model_name=model_name)

#     def embed_documents(self, documents: List) -> List[List[float]]:
#         docs = [doc if isinstance(doc, Document) else Document(page_content=str(doc)) for doc in documents]
#         texts = [doc.page_content for doc in docs]
#         return self.embedding_model.embed_documents(texts)

#     def embed_query(self, query: str) -> List[float]:
#         return self.embedding_model.embed_query(query)


import threading
from typing import List, Union
from langchain_core.documents import Document # type: ignore
from langchain_huggingface import HuggingFaceEmbeddings # type: ignore

class Embedded:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self._lock = threading.Lock()
        try:
            self.embedding_model = HuggingFaceEmbeddings(model_name=model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize embedding model: {e}")

    def embed_documents(self, documents: List[Union[Document, str]]) -> List[List[float]]:
        with self._lock:
            try:
                docs = [
                    doc if isinstance(doc, Document) else Document(page_content=str(doc)) 
                    for doc in documents
                ]
                texts = [doc.page_content for doc in docs]
                return self.embedding_model.embed_documents(texts)
            except Exception as e:
                raise RuntimeError(f"Document embedding failed: {e}")

    def embed_query(self, query: str) -> List[float]:
        with self._lock:
            try:
                if not query.strip():
                    raise ValueError("Query string is empty")
                return self.embedding_model.embed_query(query)
            except Exception as e:
                raise RuntimeError(f"Query embedding failed: {e}")


