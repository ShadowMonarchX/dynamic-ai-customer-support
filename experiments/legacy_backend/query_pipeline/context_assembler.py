from typing import Dict, Any, List, Union
from langchain_core.documents import Document


class ContextAssembler:
    def __init__(self):
        self.base_instruction = (
            "You are a customer support AI.\n"
            "Use ONLY the information provided in the knowledge base below.\n"
            "DO NOT assume, guess, or invent facts.\n"
            "If the answer is not present, say: "
            "'I don’t have that information right now.'"
        )

    def assemble(
        self,
        retrieval: Union[Dict[str, Any], List[Any]],
        intent: str = "",
        max_chars: int = 2500,
    ) -> str:
        try:
            # Handle if retrieval is just a list of docs
            if isinstance(retrieval, list):
                docs: List[Any] = retrieval
            elif isinstance(retrieval, dict):
                docs: List[Any] = retrieval.get("docs", [])
            else:
                docs = []

            if not docs:
                return f"{self.base_instruction}\n\nKNOWLEDGE BASE:\nNo relevant information available."

            context_parts = []
            current_length = 0

            for doc in docs:
                # Support Document objects
                if isinstance(doc, Document):
                    text = doc.page_content.strip()
                else:
                    text = str(doc).strip()

                if not text:
                    continue
                if current_length + len(text) > max_chars:
                    break
                context_parts.append(text)
                current_length += len(text)

            context_text = "\n\n---\n\n".join(context_parts)

            return f"{self.base_instruction}\n\nKNOWLEDGE BASE:\n{context_text}"
        except Exception:
            return f"{self.base_instruction}\n\nKNOWLEDGE BASE:\nNo relevant information available."
