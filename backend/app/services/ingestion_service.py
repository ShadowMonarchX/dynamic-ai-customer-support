from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from backend.app.core.logging import get_logger
from backend.app.core.settings import Settings


class IngestionService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = get_logger("ingestion")
        self._header_pattern = re.compile(r"^[A-Z0-9_\- ]{4,}$")

    def _derive_metadata(self, text: str) -> dict[str, Any]:
        lowered = text.lower()
        category = "general"
        topic = "general"

        if any(word in lowered for word in ("refund", "billing", "payment")):
            category = "billing"
            topic = "refund" if "refund" in lowered else "billing"
        elif any(word in lowered for word in ("password", "login", "account")):
            category = "technical"
            topic = "account"
        elif any(word in lowered for word in ("contact", "email", "phone")):
            category = "general"
            topic = "contact"
        elif any(word in lowered for word in ("who is", "profile", "biography", "bio")):
            category = "identity"
            topic = "identity"

        return {
            "category": category,
            "topic": topic,
            "content_type": "identity" if topic == "identity" else "general",
            "status": "active",
            "confidence_weight": 1.0,
        }

    def _chunk_text(self, text: str, chunk_size: int = 700, overlap: int = 120) -> list[str]:
        normalized = re.sub(r"[ \t]+", " ", text).strip()
        if not normalized:
            return []

        sections: list[str] = []
        buffer: list[str] = []
        for line in normalized.splitlines():
            stripped = line.strip()
            if self._header_pattern.match(stripped) and buffer:
                sections.append("\n".join(buffer).strip())
                buffer = [stripped]
            else:
                buffer.append(stripped)
        if buffer:
            sections.append("\n".join(buffer).strip())

        chunks: list[str] = []
        for section in sections:
            start = 0
            while start < len(section):
                end = min(len(section), start + chunk_size)
                chunk = section[start:end].strip()
                if chunk:
                    chunks.append(chunk)
                if end >= len(section):
                    break
                start = max(0, end - overlap)
        return chunks

    async def rebuild_artifacts(self) -> int:
        source_path: Path = self.settings.data_path
        raw_text = source_path.read_text(encoding="utf-8", errors="ignore")
        chunks = self._chunk_text(raw_text)

        docs: list[dict[str, Any]] = []
        for index, chunk in enumerate(chunks):
            doc_id = hashlib.sha256(f"{index}:{chunk}".encode()).hexdigest()[:16]
            docs.append(
                {
                    "doc_id": doc_id,
                    "text": chunk,
                    "source": str(source_path.name),
                    "metadata": self._derive_metadata(chunk),
                }
            )

        payload = {
            "version": 1,
            "source": str(source_path),
            "count": len(docs),
            "documents": docs,
        }
        self.settings.vector_artifact_path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        self.logger.info("Rebuilt vector artifacts", extra={"count": len(docs)})
        return len(docs)

    async def ensure_artifacts(self) -> None:
        if not self.settings.vector_artifact_path.exists():
            await self.rebuild_artifacts()
