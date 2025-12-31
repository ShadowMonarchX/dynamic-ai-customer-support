# ingestion_manager.py (Optional)
# (Pipeline Orchestration Layer)
# Purpose
#
# Coordinate the entire ingestion lifecycle.
#
# What This File Controls
#
# Order of ingestion steps
#
# Re-ingestion when data changes
#
# Scheduled or event-based refresh
#
# Error handling and validation
#
# Why It’s Optional
#
# Not required for early-stage systems
#
# Useful for enterprise-scale ingestion
#
# 📌 Think of this file as the conductor, not a processor.

import threading
from typing import List
from langchain_core.documents import Document  # type: ignore
from preprocessing import Preprocessor  # type: ignore
from embedding import Embedded  # type: ignore
from metadata_enricher import MetadataEnricher  # type: ignore

class IngestionManager:
    def __init__(
        self,
        preprocessor: Preprocessor,
        embedder: Embedded,
        metadata_enricher: MetadataEnricher = None
    ):
        self._lock = threading.Lock()
        self.preprocessor = preprocessor
        self.embedder = embedder
        self.metadata_enricher = metadata_enricher

    def ingest_documents(self, raw_documents: List[Document]):
        with self._lock:
            try:
                # Step 1: Clean, normalize, and chunk
                processed_docs = self.preprocessor.transform_documents(raw_documents)

                # Step 2: Optional metadata enrichment
                if self.metadata_enricher:
                    processed_docs = self.metadata_enricher.enrich_documents(processed_docs)

                # Step 3: Generate embeddings
                embeddings = self.embedder.embed_documents(processed_docs)

                return processed_docs, embeddings
            except Exception as e:
                raise RuntimeError(f"Ingestion pipeline failed: {e}")

    def refresh_documents(self, raw_documents: List[Document]):
        # Re-ingestion for updated or changed data
        return self.ingest_documents(raw_documents)
