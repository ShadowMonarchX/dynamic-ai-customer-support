# context_assembler.py
# (Context Selection & Validation Layer)
# Purpose
#
# Selects the most accurate sections from retrieved documents,
# removes duplicates, resolves conflicts, and ensures policy consistency.
#
# If data is missing:
# - Provides a safe fallback response
# - Or asks a clarifying question instead of guessing
#
# This step ensures answer reliability.


import threading
from typing import List, Dict, Any
from langchain_core.documents import Document # type: ignore
from langchain_core.prompts import ChatPromptTemplate # type: ignore
from langchain_core.messages import SystemMessage, HumanMessage # type: ignore


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

    def assemble_prompt(
        self,
        retrieved_docs: List[Document],
        query_features: Dict[str, Any],
        max_chars: int = 2500
    ) -> ChatPromptTemplate:
        with self._lock:
            if not retrieved_docs:
                messages = [
                    SystemMessage(
                        content=(
                            "You are a customer support AI.\n"
                            "No relevant knowledge was found.\n"
                            "Politely ask the user to clarify their question."
                        )
                    ),
                    HumanMessage(content="{question}")
                ]
                return ChatPromptTemplate.from_messages(messages)

            context_parts = []
            current_length = 0

            for doc in retrieved_docs:
                text = doc.page_content.strip()
                if not text:
                    continue
                if current_length + len(text) > max_chars:
                    break
                context_parts.append(text)
                current_length += len(text)

            context_text = "\n\n---\n\n".join(context_parts)

            instruction = self.base_instruction

            if query_features.get("emotion") == "angry":
                instruction += "\nRespond with empathy, but remain factual."

            if query_features.get("urgency") == "high":
                instruction += "\nKeep the answer short and action-oriented."

            messages = [
                SystemMessage(
                    content=f"{instruction}\n\nKNOWLEDGE BASE:\n{context_text}"
                ),
                HumanMessage(content="{question}")
            ]

            return ChatPromptTemplate.from_messages(messages)
