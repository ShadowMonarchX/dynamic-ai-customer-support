class FAQResponseStrategy:
    def generate_response(self, answer: str) -> str:
        return (
            f"{answer}\n\n"
            "If you’d like more details, just let me know."
        )
