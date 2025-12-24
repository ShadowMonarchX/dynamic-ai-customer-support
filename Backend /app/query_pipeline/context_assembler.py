class ContextAssembler:
    def __init__(self, retrieved_chunks, system_instructions=None, conversation_history=None):
        self.chunks = retrieved_chunks
        self.instructions = system_instructions or "Answer only using retrieved context."
        self.history = conversation_history or []

    def assemble(self, max_chars=4000):
        parts = []
        length = 0

        for chunk in self.chunks:
            text = " ".join(chunk) if isinstance(chunk, list) else str(chunk)
            if not text:
                continue
            if length + len(text) > max_chars:
                break
            parts.append(text)
            length += len(text)

        if self.history:
            history_text = "\n".join(map(str, self.history))
            if length + len(history_text) <= max_chars:
                parts.append("Conversation History:\n" + history_text)
                length += len(history_text)

        instructions_text = f"System Instructions:\n{self.instructions}"
        if length + len(instructions_text) <= max_chars:
            parts.append(instructions_text)

        return "\n\n".join(parts)
