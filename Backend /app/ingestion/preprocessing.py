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

# import re
# import threading
# from typing import List, Dict, Any
# from langchain_core.documents import Document  # type: ignore
# from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore


# class Preprocessor:
#     def __init__(self, chunk_size: int = 900, chunk_overlap: int = 200):
#         self._lock = threading.Lock()

#         # Slightly larger overlap to preserve identity context
#         self.splitter = RecursiveCharacterTextSplitter(
#             chunk_size=chunk_size,
#             chunk_overlap=chunk_overlap,
#             separators=["\n### ", "\n\n", ". ", "\n", " "]
#         )

#         # Remove ONLY visual noise, not content
#         self.noise_pattern = re.compile(r'-{3,}')
#         self.header_pattern = re.compile(r'(### .*)')

#         # Identity patterns (DO NOT REMOVE)
#         self.email_pattern = re.compile(r'\b[\w\.-]+@[\w\.-]+\.\w+\b')
#         self.phone_pattern = re.compile(r'\+?\d[\d\s\-]{8,}\d')
#         self.name_pattern = re.compile(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)+\b')

#     def _clean_content(self, text: str) -> str:
#         """
#         Clean text WITHOUT destroying identity signals
#         """
#         try:
#             # Remove separators like ----
#             text = self.noise_pattern.sub('', text)

#             # Normalize known variants (safe normalization)
#             replacements = {
#                 "shipping cost": "Shipping Charges",
#                 "delivery fee": "Shipping Charges",
#                 "postage": "Shipping Charges",
#             }
#             for k, v in replacements.items():
#                 text = re.sub(k, v, text, flags=re.IGNORECASE)

#             # Normalize whitespace ONLY
#             text = re.sub(r'[ \t]+', ' ', text)
#             text = re.sub(r'\n{3,}', '\n\n', text)

#             return text.strip()
#         except Exception as e:
#             raise RuntimeError(f"Cleaning Error: {e}")

#     def _derive_metadata(self, header: str) -> Dict[str, Any]:
#         meta = {
#             "category": "general",
#             "urgency": "low",
#             "content_type": "general"
#         }

#         header_upper = header.upper()

#         if "ABOUT" in header_upper or "PROFILE" in header_upper:
#             meta.update({"content_type": "identity", "topic": "profile"})
#         elif "REFUND" in header_upper:
#             meta.update({"category": "billing", "urgency": "high", "topic": "refund"})
#         elif "TECHNICAL" in header_upper or "SUPPORT" in header_upper:
#             meta.update({"category": "technical", "urgency": "medium"})
#         elif "CONTACT" in header_upper:
#             meta.update({"content_type": "contact", "topic": "contact_info"})

#         return meta

#     def _is_identity_rich(self, text: str) -> bool:
#         """
#         Detect identity-heavy content (names, emails, phones)
#         """
#         return (
#             self.email_pattern.search(text)
#             or self.phone_pattern.search(text)
#             or self.name_pattern.search(text)
#         )

#     def transform_documents(self, documents: List[Document]) -> List[Document]:
#         with self._lock:
#             processed_chunks = []

#             try:
#                 for doc in documents:
#                     raw_text = doc.page_content
#                     sections = self.header_pattern.split(raw_text)

#                     current_meta = doc.metadata.copy()

#                     for section in sections:
#                         section = section.strip()
#                         if not section:
#                             continue

#                         # Header → metadata
#                         if section.startswith("###"):
#                             current_meta.update(self._derive_metadata(section))
#                             continue

#                         cleaned_text = self._clean_content(section)
#                         if not cleaned_text:
#                             continue

#                         # 🔥 KEY FIX: identity content stays larger
#                         if self._is_identity_rich(cleaned_text):
#                             processed_chunks.append(
#                                 Document(
#                                     page_content=cleaned_text,
#                                     metadata={**current_meta, "identity_rich": True}
#                                 )
#                             )
#                             continue

#                         # Normal chunking for non-identity text
#                         chunks = self.splitter.split_text(cleaned_text)
#                         for chunk in chunks:
#                             processed_chunks.append(
#                                 Document(
#                                     page_content=chunk,
#                                     metadata=current_meta.copy()
#                                 )
#                             )

#                 return processed_chunks

#             except Exception as e:
#                 raise RuntimeError(f"Transformation Pipeline Failed: {e}")

#     def get_stats(self, docs: List[Document]) -> str:
#         return f"Total Chunks Generated: {len(docs)}"


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
            separators=["\n### ", "\n\n", ". ", "\n", " "]
        )
        self.noise_pattern = re.compile(r'-{3,}')
        self.header_pattern = re.compile(r'(### .*)')
        self.email_pattern = re.compile(r'\b[\w\.-]+@[\w\.-]+\.\w+\b')
        self.phone_pattern = re.compile(r'\+?\d[\d\s\-]{8,}\d')
        self.name_pattern = re.compile(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)+\b')

    def _clean_content(self, text: str) -> str:
        text = self.noise_pattern.sub('', text)

        replacements = {
            "shipping cost": "Shipping Charges",
            "delivery fee": "Shipping Charges",
            "postage": "Shipping Charges",
        }
        for k, v in replacements.items():
            text = re.sub(k, v, text, flags=re.IGNORECASE)

        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text.strip()

    def _derive_metadata(self, header: str) -> Dict[str, Any]:
        meta = {
            "category": "general",
            "urgency": "low",
            "content_type": "general"
        }

        h = header.upper()

        if "ABOUT" in h or "PROFILE" in h:
            meta.update({"content_type": "identity", "topic": "profile"})
        elif "REFUND" in h:
            meta.update({"category": "billing", "urgency": "high", "topic": "refund"})
        elif "TECHNICAL" in h or "SUPPORT" in h:
            meta.update({"category": "technical", "urgency": "medium"})
        elif "CONTACT" in h:
            meta.update({"content_type": "contact", "topic": "contact_info"})

        return meta

    def _is_identity_rich(self, text: str) -> bool:
        return bool(
            self.email_pattern.search(text)
            or self.phone_pattern.search(text)
            or self.name_pattern.search(text)
        )

    def transform_documents(self, documents: List[Document]) -> List[Document]:
        with self._lock:
            processed_chunks = []

            for doc in documents:
                raw_text = doc.page_content or ""
                sections = self.header_pattern.split(raw_text)
                current_meta = dict(doc.metadata)

                for section in sections:
                    section = section.strip()
                    if not section:
                        continue

                    if section.startswith("###"):
                        current_meta.update(self._derive_metadata(section))
                        continue

                    cleaned_text = self._clean_content(section)
                    if not cleaned_text:
                        continue

                    if self._is_identity_rich(cleaned_text):
                        processed_chunks.append(
                            Document(
                                page_content=cleaned_text,
                                metadata={**current_meta, "identity_rich": True}
                            )
                        )
                        continue

                    for chunk in self.splitter.split_text(cleaned_text):
                        processed_chunks.append(
                            Document(
                                page_content=chunk,
                                metadata=current_meta.copy()
                            )
                        )

            return processed_chunks

    def get_stats(self, docs: List[Document]) -> str:
        return f"Total Chunks Generated: {len(docs)}"
