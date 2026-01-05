import threading
from typing import List, Dict, Any
from langchain_core.documents import Document


class MetadataEnricher:
    def __init__(
        self,
        default_source: str = "docs",
        default_access: str = "public",
        default_version: str = "v1",
    ):
        self._lock = threading.Lock()
        self.default_source = default_source
        self.default_access = default_access
        self.default_version = default_version

    def _enrich_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        enriched = dict(metadata or {})
        enriched.setdefault("source", self.default_source)
        enriched.setdefault("access_level", self.default_access)
        enriched.setdefault("version", self.default_version)
        enriched.setdefault("category", "general")
        enriched.setdefault("topic", enriched.get("category", "general"))
        enriched.setdefault("content_type", "general")
        enriched.setdefault("urgency", "low")
        enriched.setdefault("identity_rich", False)
        enriched.setdefault("confidence_weight", 1.0)
        enriched.setdefault("status", "active")
        return enriched

    def enrich_documents(self, documents: List[Document]) -> List[Document]:
        with self._lock:
            return [
                Document(
                    page_content=doc.page_content,
                    metadata=self._enrich_metadata(doc.metadata),
                )
                for doc in documents
            ]
