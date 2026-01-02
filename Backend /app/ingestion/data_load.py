# 1. data_load.py
# (Data Collection Layer)
# Purpose

# This file is responsible for bringing all raw business data into the AI system.

# What It Handles (Conceptually)

# Reads data from multiple sources

# Acts as the single entry point for ingestion

# Abstracts away original file formats

# Data Sources Covered

# Website pages

# Help center articles

# FAQs

# Product documentation

# Order & refund policies

# App database exports

# Support tickets

# Admin dashboards

# File Types Ingested

# .html, .md, .txt

# .pdf, .docx

# .json, .csv, .xml

# Database summaries

# Ticket & chat text

# 📌 Key Principle
# All formats are converted into clean plain text before moving forward.


# import os
# import threading
# from langchain_community.document_loaders import TextLoader, DirectoryLoader # type: ignore
# from langchain_core.documents import Document # type: ignore

# class DataSource:

#     def __init__(self, path: str, chunk_size: int = 300, chunk_overlap: int = 50):
#         self.path = os.path.abspath(path)
#         self.documents: list[Document] = []
#         self._lock = threading.Lock()
#         self.chunk_size = chunk_size
#         self.chunk_overlap = chunk_overlap

#     def load_data(self) -> None:
#         with self._lock:
#             try:
#                 if not os.path.exists(self.path):
#                     raise FileNotFoundError(f"Path not found: {self.path}")

#                 raw_docs = []
#                 if os.path.isfile(self.path):
#                     if not self.path.endswith(".txt"):
#                         raise ValueError("File must be a .txt format")
#                     loader = TextLoader(self.path, encoding="utf-8")
#                     raw_docs = loader.load()

#                 elif os.path.isdir(self.path):
#                     loader = DirectoryLoader(
#                         self.path,
#                         glob="**/*.txt",
#                         loader_cls=TextLoader,
#                         loader_kwargs={"encoding": "utf-8"},
#                     )
#                     raw_docs = loader.load()
#                 else:
#                     raise TypeError("Unsupported path type")

#                 self.documents = self._split_documents(raw_docs)

#             except (FileNotFoundError, ValueError, TypeError) as e:
#                 raise RuntimeError(f"Validation Error: {e}")
#             except Exception as e:
#                 raise RuntimeError(f"Unexpected error during ingestion: {e}")

#     def _split_documents(self, docs: list[Document]) -> list[Document]:
#         chunked_docs = []
#         for doc in docs:
#             text = doc.page_content
#             start = 0
#             while start < len(text):
#                 end = min(start + self.chunk_size, len(text))
#                 chunk_text = text[start:end]
#                 chunk_doc = Document(
#                     page_content=chunk_text,
#                     metadata={**doc.metadata, "source": getattr(doc, "metadata", {}).get("source", "unknown")}
#                 )
#                 chunked_docs.append(chunk_doc)
#                 start += self.chunk_size - self.chunk_overlap
#         return chunked_docs

#     def get_documents(self) -> list[Document]:
#         with self._lock:
#             return self.documents

import os
import threading
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_core.documents import Document


class DataSource:
    def __init__(self, path: str, chunk_size: int = 300, chunk_overlap: int = 50):
        self.path = os.path.abspath(path)
        self.documents = []
        self._lock = threading.Lock()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_data(self) -> None:
        with self._lock:
            if not os.path.exists(self.path):
                raise RuntimeError(f"Path not found: {self.path}")
            if os.path.isfile(self.path):
                if not self.path.endswith(".txt"):
                    raise RuntimeError("Only .txt files are supported")
                loader = TextLoader(self.path, encoding="utf-8")
                raw_docs = loader.load()
            elif os.path.isdir(self.path):
                loader = DirectoryLoader(
                    self.path,
                    glob="**/*.txt",
                    loader_cls=TextLoader,
                    loader_kwargs={"encoding": "utf-8"},
                )
                raw_docs = loader.load()
            else:
                raise RuntimeError("Invalid path type")
            self.documents = self._split_documents(raw_docs)

    def _split_documents(self, docs):
        chunked_docs = []
        for doc in docs:
            text = doc.page_content
            start = 0
            while start < len(text):
                end = min(start + self.chunk_size, len(text))
                chunked_docs.append(
                    Document(
                        page_content=text[start:end], metadata=dict(doc.metadata or {})
                    )
                )
                start += self.chunk_size - self.chunk_overlap
        return chunked_docs

    def get_documents(self):
        with self._lock:
            return list(self.documents)
