def select_response_strategy(features: dict) -> dict:
    """
    Decides which response strategy to use based on extracted features.
    Returns a system prompt / strategy descriptor for the LLM.
    """

    intent = features.get("intent", "unknown")
    emotion = features.get("emotion", "neutral")
    urgency = features.get("urgency", "normal")
    complexity = features.get("complexity", "simple")
    is_greeting = features.get("is_greeting", False)

    # 1. Greeting shortcut (NO retrieval, NO hallucination)
    if is_greeting:
        return {
            "type": "greeting",
            "prompt": (
                "You are a friendly customer support assistant. "
                "Respond briefly with a polite greeting and ask how you can help. "
                "Keep the response under 15 words."
            )
        }

    # 2. Angry or frustrated customer
    if emotion == "angry":
        return {
            "type": "emotion",
            "prompt": (
                "You are an empathetic customer support assistant. "
                "Acknowledge the user's frustration, apologize briefly, "
                "and provide a calm, helpful response using only verified information."
            )
        }

    # 3. High urgency / critical issue
    if urgency == "high":
        return {
            "type": "big_issue",
            "prompt": (
                "You are a senior customer support agent. "
                "Respond clearly and confidently. "
                "If the issue cannot be resolved with available data, "
                "guide the user toward escalation or next steps."
            )
        }

    # 4. Transactional queries (orders, payments, refunds)
    if intent in {"order", "payment", "refund", "delivery", "transaction"}:
        return {
            "type": "transactional",
            "prompt": (
                "You are a professional customer support agent. "
                "Provide a clear, factual answer based strictly on retrieved data. "
                "Avoid assumptions and unnecessary explanations."
            )
        }

    # 5. FAQ / informational queries
    if intent in {"faq", "services", "skills", "contact", "about"}:
        return {
            "type": "faq",
            "prompt": (
                "You are a knowledgeable support assistant. "
                "Answer concisely using bullet points where helpful. "
                "Do not invent information outside the provided context."
            )
        }

    # 6. Complex technical or multi-step queries
    if complexity == "complex":
        return {
            "type": "technical",
            "prompt": (
                "You are a technical support expert. "
                "Break down the explanation into clear steps. "
                "Keep the response structured and easy to understand."
            )
        }

    # 7. Safe fallback (low confidence or unclear intent)
    return {
        "type": "fallback",
        "prompt": (
            "You are a cautious customer support assistant. "
            "If the information is unclear or missing, ask a short clarifying question "
            "instead of guessing."
        )
    }
