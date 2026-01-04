import threading
from typing import List, Tuple
from langchain_core.documents import Document
from .preprocessing import Preprocessor
from .embedding import Embedder
from .metadata_enricher import MetadataEnricher


class IngestionManager:
    def __init__(
        self,
        preprocessor: Preprocessor,
        embedder: Embedder,
        metadata_enricher: MetadataEnricher = None,
    ):
        self._lock = threading.Lock()
        self.preprocessor = preprocessor
        self.embedder = embedder
        self.metadata_enricher = metadata_enricher

    def ingest_documents(
        self, raw_documents: List[Document]
    ) -> Tuple[List[Document], List[List[float]]]:
        with self._lock:
            processed_docs = []
            for doc in raw_documents:
                try:
                    chunks = self.preprocessor.transform_documents([doc])
                    if self.metadata_enricher:
                        chunks = self.metadata_enricher.enrich_documents(chunks)
                    processed_docs.extend(chunks)
                except Exception:
                    continue
            if not processed_docs:
                raise RuntimeError("No valid documents to ingest")
            embeddings = []
            for doc in processed_docs:
                try:
                    vec = self.embedder.embed_documents([doc])
                    if vec:
                        embeddings.extend(vec)
                except Exception:
                    continue
            if not embeddings:
                raise RuntimeError("Embedding generation failed")
            return processed_docs, embeddings

    def refresh_documents(self, raw_documents: List[Document]):
        return self.ingest_documents(raw_documents)
