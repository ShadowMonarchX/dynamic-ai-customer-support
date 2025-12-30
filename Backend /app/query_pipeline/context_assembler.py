import threading
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document # type: ignore
from langchain_core.prompts import ChatPromptTemplate # type: ignore
from langchain_core.messages import SystemMessage, HumanMessage # type: ignore

class ContextAssembler:
    def __init__(self):
        self._lock = threading.Lock()
        self.default_instruction = "Answer based on the context provided."

    def assemble_prompt(
        self,
        retrieved_docs: List[Document],
        query_features: Dict[str, Any],
        max_chars: int = 4000
    ) -> ChatPromptTemplate:
        with self._lock:
            try:
                context_parts = []
                current_length = 0

                for doc in retrieved_docs:
                    text = doc.page_content.strip()
                    if total_len := (current_length + len(text)) > max_chars:
                        break
                    context_parts.append(text)
                    current_length += len(text)

                context_text = "\n\n".join(context_parts)
                
                # Dynamic Instruction based on engineered features
                instruction = self.default_instruction
                if query_features.get("emotion") == "frustrated":
                    instruction += " The user is frustrated; be empathetic and apologetic."
                if query_features.get("urgency") == "high":
                    instruction += " Provide a concise, immediate solution."

                messages = [
                    SystemMessage(content=f"{instruction}\n\nKNOWLEDGE BASE:\n{context_text}"),
                    HumanMessage(content="{question}")
                ]

                return ChatPromptTemplate.from_messages(messages)
                
            except Exception as e:
                raise RuntimeError(f"Context Assembly Failed: {e}")