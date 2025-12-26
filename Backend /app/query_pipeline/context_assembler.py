from typing import List, Optional
from langchain_core.documents import Document # type: ignore
from langchain_core.prompts import ChatPromptTemplate # type: ignore
from langchain_core.messages import SystemMessage, HumanMessage # type: ignore

class ContextAssembler:
    def __init__(self, system_instructions: str = "Answer only using the retrieved context."):
        self.system_instructions = system_instructions

    def assemble_prompt(
        self,
        retrieved_docs: List[Document],
        conversation_history: Optional[List] = None,
        max_chars: int = 4000
    ) -> ChatPromptTemplate:
        context_parts = []
        total_length = 0

        for doc in retrieved_docs:
            text = doc.page_content.strip()
            if not text:
                continue
            if total_length + len(text) > max_chars:
                break
            context_parts.append(text)
            total_length += len(text)

        context_text = "\n\n".join(context_parts)

        messages = [
            SystemMessage(content=f"{self.system_instructions}\n\nContext:\n{context_text}"),
            HumanMessage(content="{question}")  # only the user's question
        ]

        return ChatPromptTemplate.from_messages(messages)
