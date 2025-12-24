import re

class QueryPreprocessor:
    def __init__(self, query):
        self.query = query

    def preprocess(self):
        q = self.query.lower()
        q = re.sub(r"[^a-zA-Z0-9\s]", "", q)
        q = re.sub(r"\s+", " ", q).strip()
        return q
