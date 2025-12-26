import re
from langchain_core.runnables import Runnable # type: ignore

class QueryPreprocessor(Runnable):
    def invoke(self, query: str) -> str:
        cleaned_query = re.sub(r"[^a-z0-9\s]", "", query.lower())
        cleaned_query = re.sub(r"\s+", " ", cleaned_query).strip()
        return cleaned_query
