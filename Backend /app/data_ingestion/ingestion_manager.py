import threading
from typing import List, Tuple
import numpy as np
from langchain_core.documents import Document

from .preprocessing import Preprocessor
from .embedding import Embedder
from .metadata_enricher import MetadataEnricher


class IngestionManager:
    def __init__(
        self,
        preprocessor: Preprocessor,
        embedder: Embedder,
        metadata_enricher: MetadataEnricher | None = None,
    ):
        self._lock = threading.Lock()
        self.preprocessor = preprocessor
        self.embedder = embedder
        self.metadata_enricher = metadata_enricher

    def ingest_documents(
        self, raw_documents: List[Document]
    ) -> Tuple[List[Document], List[np.ndarray]]:
        with self._lock:
            processed_docs: List[Document] = []

            for doc in raw_documents:
                chunks = self.preprocessor.transform_documents([doc])
                if self.metadata_enricher:
                    chunks = self.metadata_enricher.enrich_documents(chunks)
                processed_docs.extend(chunks)

            if not processed_docs:
                raise RuntimeError("No valid documents to ingest")

            embeddings_array = self.embedder.embed_documents(processed_docs)

            if (
                embeddings_array is None
                or not isinstance(embeddings_array, np.ndarray)
                or embeddings_array.ndim != 2
                or embeddings_array.shape[0] == 0
            ):
                raise RuntimeError("Embedding generation failed")

            embeddings = [embeddings_array[i] for i in range(embeddings_array.shape[0])]

            return processed_docs, embeddings

    def refresh_documents(self, raw_documents: List[Document]):
        return self.ingest_documents(raw_documents)
