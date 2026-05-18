from __future__ import annotations

import re

from backend.app.core.exceptions import ValidationFailure


class SafetyService:
    _jailbreak_patterns = [
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"ignore\s+previous\s+instructions",
            r"reveal\s+system\s+prompt",
            r"bypass\s+safety",
            r"jailbreak",
            r"act\s+as\s+developer",
            r"print\s+all\s+secrets",
        )
    ]

    _unsafe_output_patterns = [
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"password\s*[:=]",
            r"api[_\s-]?key\s*[:=]",
            r"secret\s*[:=]",
            r"token\s*[:=]",
        )
    ]

    def sanitize_query(self, query: str) -> str:
        cleaned = re.sub(r"\s+", " ", query).strip()
        if not cleaned:
            raise ValidationFailure("Query cannot be empty")
        if len(cleaned) > 4000:
            raise ValidationFailure("Query exceeds size limit")
        self.enforce_input_policy(cleaned)
        return cleaned

    def enforce_input_policy(self, query: str) -> None:
        for pattern in self._jailbreak_patterns:
            if pattern.search(query):
                raise ValidationFailure("Unsafe prompt detected")

    def sanitize_context(self, context: str) -> str:
        # Instruction isolation: retrieved context is strictly delimited.
        return f"<retrieved_context>\n{context}\n</retrieved_context>"

    def moderate_output(self, output: str) -> str:
        lowered = output.lower()
        for pattern in self._unsafe_output_patterns:
            if pattern.search(lowered):
                return "I cannot provide sensitive information. Please contact support securely."
        return output
