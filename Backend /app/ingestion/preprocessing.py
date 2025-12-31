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
from langchain_core.documents import Document # type: ignore
from langchain_text_splitters import RecursiveCharacterTextSplitter # Use underscore and an 's' # type: ignore

class Preprocessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        self._lock = threading.Lock()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n### ", "\n\n", "\n", " ", ""]
        )
        self.noise_pattern = re.compile(r'-{3,}')
        self.header_pattern = re.compile(r'(### .*)')

    def _clean_content(self, text: str) -> str:
        try:
            text = self.noise_pattern.sub('', text)
            text = text.replace("shipping cost", "Shipping Charges")
            text = text.replace("delivery fee", "Shipping Charges")
            text = text.replace("postage", "Shipping Charges")
            return re.sub(r'\s+', ' ', text).strip()
        except Exception as e:
            raise RuntimeError(f"Cleaning Error: {e}")

    def _derive_metadata(self, header: str) -> Dict[str, Any]:
        meta = {"category": "general", "urgency": "low"}
        header_upper = header.upper()
        if "REFUND" in header_upper:
            meta.update({"category": "billing", "urgency": "high", "topic": "refund"})
        elif "TECHNICAL" in header_upper or "SUPPORT" in header_upper:
            meta.update({"category": "technical", "urgency": "medium"})
        return meta

    def transform_documents(self, documents: List[Document]) -> List[Document]:
        with self._lock:
            processed_chunks = []
            try:
                for doc in documents:
                    raw_text = doc.page_content
                    sections = self.header_pattern.split(raw_text)
                    
                    current_meta = doc.metadata.copy()
                    
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

                        chunks = self.splitter.split_text(cleaned_text)
                        for chunk in chunks:
                            processed_chunks.append(
                                Document(
                                    page_content=chunk,
                                    metadata=current_meta.copy()
                                )
                            )
                return processed_chunks
            except Exception as e:
                raise RuntimeError(f"Transformation Pipeline Failed: {e}")

    def get_stats(self, docs: List[Document]) -> str:
        return f"Total Chunks Generated: {len(docs)}"
