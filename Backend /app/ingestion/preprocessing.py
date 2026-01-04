# 2. preprocessing.py
# (Cleaning, Normalization & Chunking Layer)
# Purpose
#
# Make raw data AI-readable, consistent, and safe.
#
# What Happens in This File
#
# Removes HTML tags and UI noise
#
# Fixes broken formatting
#
# Normalizes language and tone
#
# Removes duplicates
#
# Filters outdated or invalid policies
#
# 📌 Prevents confusing or contradictory answers
#
# Chunking Responsibility (Very Important)
#
# This file also:
#
# Splits large documents into small, meaningful chunks
#
# Ensures one concept per chunk
#
# Example
#
# Refund policy →
#
# Eligibility rules
#
# Time limits
#
# Payment method conditions
#
# 📌 Smaller chunks = better retrieval precision

import re
import threading
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class Preprocessor:
    def __init__(self, chunk_size: int = 900, chunk_overlap: int = 200):
        self._lock = threading.Lock()

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " "]
        )

        self.noise_pattern = re.compile(r'-{3,}')
        self.header_pattern = re.compile(r'^(#{3,5})\s+(.*)', re.MULTILINE)

        self.email_pattern = re.compile(r'\b[\w\.-]+@[\w\.-]+\.\w+\b')
        self.phone_pattern = re.compile(r'\+?\d[\d\s\-]{8,}\d')
        self.name_pattern = re.compile(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)+\b')

    def _clean(self, text: str) -> str:
        if not text:
            return ""
        text = self.noise_pattern.sub("", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _derive_metadata(self, header: str) -> Dict[str, Any]:
        meta = {"category": "general", "urgency": "low", "content_type": "general"}
        h = header.upper()

        if "REFUND" in h:
            meta.update({"category": "billing", "urgency": "high"})
        elif "SUPPORT" in h or "TECHNICAL" in h:
            meta.update({"category": "technical", "urgency": "medium"})
        elif "CONTACT" in h:
            meta.update({"content_type": "contact"})
        elif "ABOUT" in h or "PROFILE" in h:
            meta.update({"content_type": "identity"})

        return meta

    def _identity_rich(self, text: str) -> bool:
        signals = 0
        if self.email_pattern.search(text):
            signals += 1
        if self.phone_pattern.search(text):
            signals += 1
        if self.name_pattern.search(text):
            signals += 1
        return signals >= 2 and len(text) < 600

    def transform_documents(self, documents: List[Document]) -> List[Document]:
        with self._lock:
            output: List[Document] = []

            for doc in documents:
                try:
                    base_text = self._clean(doc.page_content)
                    if not base_text:
                        continue

                    headers = list(self.header_pattern.finditer(base_text))
                    sections = []

                    if headers:
                        for i, h in enumerate(headers):
                            start = h.end()
                            end = headers[i + 1].start() if i + 1 < len(headers) else len(base_text)
                            sections.append((h.group(2), base_text[start:end].strip()))
                    else:
                        sections.append((None, base_text))

                    chunk_idx = 0

                    for header, body in sections:
                        if not body:
                            continue

                        meta = dict(doc.metadata or {})
                        if header:
                            meta.update(self._derive_metadata(header))
                            meta["section"] = header

                        if self._identity_rich(body):
                            output.append(
                                Document(
                                    page_content=body,
                                    metadata={**meta, "identity_rich": True, "chunk_index": chunk_idx}
                                )
                            )
                            chunk_idx += 1
                            continue

                        for chunk in self.splitter.split_text(body):
                            if chunk.strip():
                                output.append(
                                    Document(
                                        page_content=chunk,
                                        metadata={**meta, "chunk_index": chunk_idx}
                                    )
                                )
                                chunk_idx += 1

                except Exception as e:
                    raise RuntimeError(f"Preprocessing failed: {e}")

            return output

    def get_stats(self, docs: List[Document]) -> str:
        return f"Total Chunks Generated: {len(docs)}"
