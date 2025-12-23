from preprocessing import Preprocessor, DataSource
from sentence_transformers import SentenceTransformer

# Load text data
source = DataSource("training_data.txt")
source.load_data()
texts = source.get_data()

# Preprocess
processor = Preprocessor(texts)
processor.preprocess()
processed_texts = processor.get_processed()  # <-- use get_processed(), not get_data()

class Embedded:
    def __init__(self, texts, model_name="all-MiniLM-L6-v2"):
        self.texts = texts
        self.model = SentenceTransformer(model_name)
        self.vectors = []

    def generate_embeddings(self):
        self.vectors = [self.model.encode(" ".join(text)) for text in self.texts]

    def get_embeddings(self):
        return self.vectors

embedded = Embedded(processed_texts)
embedded.generate_embeddings()
vectors = embedded.get_embeddings()


print("\n")
print("----------------------------------")
print("Generated Embeddings:")
print("----------------------------------")
print("\n")
for v in vectors:
    print(v)
