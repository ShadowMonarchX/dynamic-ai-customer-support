import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_ingestion_pipeline.preprocessing import Preprocessor
from data_ingestion_pipeline.data_load import DataSource
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

data_path = '/Users/jenishshekhada/Desktop/Inten/dynamic-ai-customer-support/backend /data/training_data.txt'

if not os.path.exists(data_path):
    raise FileNotFoundError(f"training_data.txt not found at: {data_path}")

source = DataSource(data_path)
source.load_data()
texts = source.get_data()

processor = Preprocessor(texts)
processor.preprocess()
processed_texts = processor.get_processed()

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = [model.encode(" ".join(text)) for text in processed_texts]
embedding_dim = embeddings[0].shape[0]

index = faiss.IndexFlatL2(embedding_dim)
index.add(np.array(embeddings).astype("float32"))

query = "Backend API design"
query_vec = model.encode(query).astype("float32").reshape(1, -1)
D, I = index.search(query_vec, k=3)


print("\n")
print("----------------------------------")
print("Top similar documents for query:", query)
print("----------------------------------")
print(I)
