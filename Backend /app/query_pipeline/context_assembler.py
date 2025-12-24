class ContextAssembler:
    def __init__(self, retrieved_chunks, system_instructions=None, conversation_history=None):
        self.chunks = retrieved_chunks
        self.instructions = system_instructions or "Answer only using retrieved context."
        self.history = conversation_history or []

    def assemble(self):
        context = "\n".join(self.chunks)
        if self.history:
            context += "\nConversation History:\n" + "\n".join(self.history)
        context += f"\nSystem Instructions:\n{self.instructions}"
        return context
