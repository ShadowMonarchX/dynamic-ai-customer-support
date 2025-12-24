class QueryEmbedder:
    def __init__(self, model):
        self.model = model

    def embed(self, query_text):
        return self.model.encode(query_text)
