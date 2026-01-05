from pathlib import Path
from typing import List
from langchain_core.documents import Document# type: ignore


class DataSource:
    def __init__(self, path: str):
        self.path = Path(path)
        self._documents: List[Document] = []

    def load(self) -> List[Document]:
        if not self.path.exists():
            raise FileNotFoundError(f"Data file not found: {self.path}")

        text = self.path.read_text(encoding="utf-8", errors="ignore")

        self._documents = [
            Document(
                page_content=text,
                metadata={
                    "source": str(self.path),
                    "category": "general",
                    "urgency": "low",
                    "content_type": "general",
                },
            )
        ]
        return self._documents
