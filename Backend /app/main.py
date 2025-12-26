# from app.ingestion.data_load import DataSource
# from app.ingestion.preprocessing import Preprocessor
# from app.ingestion.embedding import Embedded
# from app.vector_store.faiss_index import FAISSIndex
# from app.query_pipeline.query_preprocess import QueryPreprocessor
# from app.query_pipeline.query_embed import QueryEmbedder
# from app.query_pipeline.context_assembler import ContextAssembler
# from app.reasoning.llm_reasoner import LLMReasoner
# from app.validation.answer_validator import AnswerValidator
# from langchain_core.documents import Document # type: ignore
# import numpy as np # type: ignore
# import os

# data_path = '/Users/jenishshekhada/Desktop/Inten/dynamic-ai-customer-support/backend /data/training_data.txt'

# if not os.path.exists(data_path):
#     raise FileNotFoundError(data_path)

# source = DataSource(data_path)
# source.load_data()
# raw_texts = source.get_documents()
# if not raw_texts:
#     raise ValueError("No documents loaded")

# processor = Preprocessor()
# processed_texts = processor.transform_documents(raw_texts)
# documents = [Document(page_content=text) for text in processed_texts]

# embedder = Embedded()
# doc_vectors = embedder.embed_documents(documents)
# doc_vectors = np.array(doc_vectors, dtype="float32")
# if doc_vectors.ndim == 1:
#     doc_vectors = doc_vectors.reshape(1, -1)

# faiss_index = FAISSIndex(doc_vectors)

# user_query = "Give me contact details for Nayan Raval"

# query_processor = QueryPreprocessor()
# clean_query = query_processor.invoke(user_query)

# query_vector = embedder.embed_query(clean_query)
# query_vector = np.array([query_vector], dtype="float32")
# if query_vector.ndim == 1:
#     query_vector = query_vector.reshape(1, -1)

# D, I = faiss_index.similarity_search(query_vector, top_k=3)
# retrieved_chunks = [documents[i] for i in I[0] if i < len(documents)]

# assembler = ContextAssembler(
#     system_instructions="Answer only using the retrieved context. Do not hallucinate."
# )
# prompt = assembler.assemble_prompt(retrieved_docs=retrieved_chunks)
# context = prompt.format_prompt(question=user_query).to_string()

# reasoner = LLMReasoner(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
# answer = reasoner.invoke({"query": user_query, "context": context})

# validator = AnswerValidator()
# validation = validator.invoke({"answer": answer, "context": context})

# print("----------------------")
# print("User Query:", user_query)
# print("----------------------")
# print("\nAssembled Context:\n")
# print(context)
# print("\n----------------------")
# print("Final Answer:\n")
# print(answer)
# print("\nConfidence:", validation["confidence"])
# print("Issues:", validation["issues"])



import os
import numpy as np

from app.ingestion.data_load import DataSource
from app.ingestion.preprocessing import Preprocessor
from app.ingestion.embedding import Embedded
from app.vector_store.faiss_index import FAISSIndex
from app.query_pipeline.query_preprocess import QueryPreprocessor
from app.query_pipeline.context_assembler import ContextAssembler
from app.reasoning.llm_reasoner import LLMReasoner
from app.validation.answer_validator import AnswerValidator
from langchain_core.documents import Document  # make sure installed


# -------------------- Load Data --------------------
data_path = '/Users/jenishshekhada/Desktop/Inten/dynamic-ai-customer-support/backend /data/training_data.txt'
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Data file not found: {data_path}")

source = DataSource(data_path)
source.load_data()
raw_texts = source.get_documents()

if not raw_texts:
    raise ValueError("No documents found in the data file")


# -------------------- Preprocess --------------------
processor = Preprocessor()
processed_texts = processor.transform_documents(raw_texts)

# Convert strings to Document objects
documents = []
for text in processed_texts:
    if isinstance(text, str):
        documents.append(Document(page_content=text))
    elif hasattr(text, "page_content"):
        documents.append(text)
    else:
        continue

if not documents:
    raise ValueError("No valid documents after preprocessing")


# -------------------- Embedding --------------------
embedder = Embedded(model_name="sentence-transformers/all-MiniLM-L6-v2")
doc_vectors = embedder.embed_documents(documents)
doc_vectors = np.array(doc_vectors, dtype="float32")
if doc_vectors.ndim == 1:
    doc_vectors = doc_vectors.reshape(1, -1)

faiss_index = FAISSIndex(doc_vectors)


# -------------------- User Query --------------------
user_query = "Give me contact details for Nayan Raval"

query_processor = QueryPreprocessor()
clean_query = query_processor.invoke(user_query)

query_vector = embedder.embed_query(clean_query)
query_vector = np.array([query_vector], dtype="float32")


# -------------------- Retrieve Relevant Chunks --------------------
D, I = faiss_index.similarity_search(query_vector, top_k=3)
retrieved_chunks = [documents[i] for i in I[0] if i < len(documents)]


# -------------------- Assemble Context --------------------
assembler = ContextAssembler(
    system_instructions="Answer only using the retrieved context. Do not hallucinate."
)
prompt = assembler.assemble_prompt(retrieved_docs=retrieved_chunks)

context = prompt.format_prompt(question=user_query).to_string()


# -------------------- Reasoning --------------------
reasoner = LLMReasoner(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
answer = reasoner.invoke({"query": user_query, "context": context})


# -------------------- Validation --------------------
validator = AnswerValidator()
validation = validator.invoke({"answer": answer, "context": context})


# -------------------- Output --------------------
print("----------------------")
print("User Query:", user_query)
print("----------------------")
print("\nAssembled Context:\n")
print(context)
print("\n----------------------")
print("Final Answer:")
print("\n----------------------")
print(answer)
print("\nConfidence:", validation["confidence"])
print("Issues:", validation["issues"])
