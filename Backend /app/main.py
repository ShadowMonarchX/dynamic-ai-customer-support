import os
from app.ingestion.data_load import DataSource
from app.ingestion.preprocessing import Preprocessor
from app.ingestion.embedding import Embedded
from app.vector_store.faiss_index import FAISSIndex
from app.query_pipeline.query_preprocess import QueryPreprocessor
from app.query_pipeline.query_embed import QueryEmbedder
from app.query_pipeline.context_assembler import ContextAssembler


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "data", "training_data.txt")

source = DataSource(data_path)
source.load_data()
texts = source.get_data()

processor = Preprocessor(texts)
processor.preprocess()
processed_texts = processor.get_processed()


embedded = Embedded(processed_texts)
embedded.generate_embeddings()
embeddings = embedded.get_embeddings()


faiss_index = FAISSIndex(embeddings)


user_query = "Is Laptop X available and what is the delivery time?"


query_proc = QueryPreprocessor(user_query)
preprocessed_query = query_proc.preprocess()


query_embedder = QueryEmbedder(embedded.model)
query_vec = query_embedder.embed(preprocessed_query)


D, I = faiss_index.search(query_vec, top_k=3)
retrieved_chunks = [processed_texts[i] for i in I[0]]


assembler = ContextAssembler(retrieved_chunks)
context = assembler.assemble()


print("User Query:", user_query)
print("\nRetrieved Chunks:")
for chunk in retrieved_chunks:
    print("-", chunk)
print("\nAssembled Context:\n", context)
