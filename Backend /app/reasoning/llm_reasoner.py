class LLMReasoner:
    def __init__(self, llm_client):
        self.llm = llm_client

    def generate_answer(self, context, user_query):
        prompt = f"""
You are an AI assistant.

RULES:
- Use ONLY the provided context
- Do NOT use external knowledge
- If context is insufficient, say so clearly
- Be concise and factual

CONTEXT:
{context}

USER QUESTION:
{user_query}

ANSWER:
"""
        response = self.llm(prompt)
        return response.strip()