import re


class QueryPreprocessor:
    def __init__(self, query):
        self.query = query

    def preprocess(self):
        return re.sub(
            r"\s+",
            " ",
            re.sub(r"[^a-z0-9\s]", "", self.query.lower()),
        ).strip()
