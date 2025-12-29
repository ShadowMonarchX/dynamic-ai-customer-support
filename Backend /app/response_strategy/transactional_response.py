class TransactionalResponseStrategy:
    def generate_response(self, action_hint: str) -> str:
        return (
            f"{action_hint}\n"
            "Please provide your order or reference number to continue."
        )
