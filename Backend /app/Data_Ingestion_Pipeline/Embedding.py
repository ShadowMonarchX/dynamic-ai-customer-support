from preprocessing import Preprocessor, DataSource
from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, texts, model_name="all-MiniLM-L6-v2"):
        self.texts = texts
        self.model = SentenceTransformer(model_name)
        self.vectors = []

    def generate_embeddings(self):
        self.vectors = [self.model.encode(" ".join(text)) for text in self.texts]

    def get_embeddings(self):
        return self.vectors


source = DataSource("data")
source.load_data()
texts = source.get_data()

processor = Preprocessor(texts)
processor.preprocess()
processed_texts = processor.get_processed()

embedder = Embedder(processed_texts)
embedder.generate_embeddings()
vectors = embedder.get_embeddings()

for v in vectors:
    print(v)
