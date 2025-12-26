from typing import List
from langchain_huggingface import HuggingFaceEmbeddings # type: ignore
from langchain_core.documents import Document # type: ignore

class Embedded:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model = HuggingFaceEmbeddings(model_name=model_name)

    def embed_documents(self, documents: List[Document]) -> List[List[float]]:
        texts = [doc.page_content for doc in documents]
        return self.embedding_model.embed_documents(texts)

    def embed_query(self, query: str) -> List[float]:
        return self.embedding_model.embed_query(query)

        return self.embedding_model.embed_query(query)


    
# from .data_load import DataSource
# from .preprocessing import Preprocessor

# # Load text data
# source = DataSource('/Users/jenishshekhada/Desktop/Inten/dynamic-ai-customer-support/backend /data/training_data.txt')
# source.load_data()
# texts = source.get_data()

# # Preprocess
# processor = Preprocessor(texts)
# processor.preprocess()
# processed_texts = processor.get_processed() 


# embedded = Embedded(processed_texts)
# embedded.generate_embeddings()
# vectors = embedded.get_embeddings()


# print("\n")
# print("----------------------------------")
# print("Generated Embeddings:")
# print("----------------------------------")
# print("\n")

# for v in vectors:
#     print(v)
