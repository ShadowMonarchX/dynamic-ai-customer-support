import re
from langchain_core.runnables import Runnable # type: ignore


class QueryPreprocessor(Runnable):
    """
    LangChain-style query preprocessor:
    - Lowercase
    - Remove special characters
    - Normalize whitespace
    """

    def invoke(self, query: str) -> str:
        return re.sub(
            r"\s+",
            " ",
            re.sub(r"[^a-z0-9\s]", "", query.lower()),
        ).strip()
