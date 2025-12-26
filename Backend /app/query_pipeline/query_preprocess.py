import re
from langchain_core.runnables import Runnable # type: ignore

class QueryPreprocessor(Runnable):
    def invoke(self, query: str) -> str:
        # Lowercase, remove special characters, normalize whitespace
        return re.sub(
            r"\s+", " ", re.sub(r"[^a-z0-9\s]", "", query.lower())
        ).strip()
