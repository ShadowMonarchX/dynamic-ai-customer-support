import os
import numpy as np
import faiss
from ..data_ingestion_pipeline import DataSource, Preprocessor, Embedded

data_path = '/Users/jenishshekhada/Desktop/Inten/dynamic-ai-customer-support/backend /data/training_data.txt'
if not os.path.exists(data_path):
    raise FileNotFoundError(data_path)

source = DataSource(data_path)
source.load_data()
texts = source.get_data()

processor = Preprocessor(texts)
processor.preprocess()
processed_texts = processor.get_processed()

embedded = Embedded(processed_texts)
embedded.generate_embeddings()
embeddings = embedded.get_embeddings()

embedding_dim = embeddings[0].shape[0]

index = faiss.IndexFlatL2(embedding_dim)
index.add(np.array(embeddings).astype("float32"))

query = "Backend API design"
query_vec = embedded.model.encode(query).astype("float32").reshape(1, -1)

D, I = index.search(query_vec, 3)

print(I)
print(D)
# print("---------------------")
# print("Top 3 similar texts:")
# print("---------------------")
# for i in I[0]:
#     print(processed_texts[i])   

