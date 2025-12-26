class QueryEmbedder:
    def __init__(self, model):
        self.model = model

    def embed(self, query_text: str):
        return self.model.embed_query(query_text)
