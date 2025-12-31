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

import os
import threading
from langchain_community.document_loaders import TextLoader, DirectoryLoader # type: ignore
from langchain_core.documents import Document # type: ignore

class DataSource:
    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        self.documents: list[Document] = []
        self._lock = threading.Lock()

    def load_data(self) -> None:
        with self._lock:
            try:
                if not os.path.exists(self.path):
                    raise FileNotFoundError(f"Path not found: {self.path}")

                if os.path.isfile(self.path):
                    if not self.path.endswith(".txt"):
                        raise ValueError("File must be a .txt format")
                    loader = TextLoader(self.path, encoding="utf-8")
                    self.documents = loader.load()

                elif os.path.isdir(self.path):
                    loader = DirectoryLoader(
                        self.path,
                        glob="**/*.txt",
                        loader_cls=TextLoader,
                        loader_kwargs={"encoding": "utf-8"},
                    )
                    self.documents = loader.load()
                else:
                    raise TypeError("Unsupported path type")

            except (FileNotFoundError, ValueError, TypeError) as e:
                raise RuntimeError(f"Validation Error: {e}")
            except Exception as e:
                raise RuntimeError(f"Unexpected error during ingestion: {e}")

    def get_documents(self) -> list[Document]:
        with self._lock:
            return self.documents