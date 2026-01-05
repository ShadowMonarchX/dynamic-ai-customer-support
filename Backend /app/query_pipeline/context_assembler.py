import threading
from typing import Dict, Any, List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
import logging

logger = logging.getLogger("ContextAssembler")
logging.basicConfig(level=logging.INFO)


class ContextAssembler:
    def __init__(self):
        self._lock = threading.Lock()
        self.base_instruction = (
            "You are a customer support AI.\n"
            "Use ONLY the information provided in the knowledge base below.\n"
            "DO NOT assume, guess, or invent facts.\n"
            "If the answer is not present, say: "
            "'I don’t have that information right now.'"
        )

    def assemble(
        self,
        retrieval: Dict[str, Any],
        intent: str,
        max_chars: int = 2500,
    ) -> str:
        with self._lock:
            docs: List[Any] = retrieval.get("docs", [])

            if not docs:
                return ""

            context_parts = []
            current_length = 0

            for doc in docs:
                text = doc.strip() if isinstance(doc, str) else str(doc).strip()
                if not text:
                    continue
                if current_length + len(text) > max_chars:
                    break
                context_parts.append(text)
                current_length += len(text)

            context_text = "\n\n---\n\n".join(context_parts)

            logger.info(f"Chunks assembled: {len(context_parts)}")

            return f"{self.base_instruction}\n\nKNOWLEDGE BASE:\n{context_text}"
