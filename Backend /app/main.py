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
from app.reasoning.llm_reasoner import LLMReasoner
from app.validation.answer_validator import AnswerValidator
from langchain_core.documents import Document
from transformers import AutoTokenizer
import re

data_path = '/Users/jenishshekhada/Desktop/Inten/dynamic-ai-customer-support/backend /data/training_data.txt'

if not os.path.exists(data_path):
    raise FileNotFoundError(f"Data file not found: {data_path}")

source = DataSource(data_path)
source.load_data()
documents = source.get_documents()

if not documents:
    raise ValueError("No documents loaded from the file.")

processor = Preprocessor()
processed_docs = processor.transform_documents(documents)

if not processed_docs:
    raise ValueError("No processed documents available.")

def split_text_into_chunks(text, tokenizer, max_tokens=500):
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = ""
    current_tokens = 0
    for sent in sentences:
        sent_tokens = len(tokenizer(sent, return_tensors="pt")["input_ids"][0])
        if current_tokens + sent_tokens > max_tokens:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sent
            current_tokens = sent_tokens
        else:
            current_chunk += " " + sent
            current_tokens += sent_tokens
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

tokenizer_for_chunks = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
chunked_texts = []
for doc in processed_docs:
    text = doc.page_content if isinstance(doc, Document) else str(doc)
    chunked_texts.extend(split_text_into_chunks(text, tokenizer_for_chunks, max_tokens=500))

embedder = Embedded(model_name="sentence-transformers/all-MiniLM-L6-v2")
doc_vectors = embedder.embed_documents([Document(page_content=chunk) for chunk in chunked_texts])
doc_vectors = np.atleast_2d(np.array(doc_vectors, dtype="float32"))
faiss_index = FAISSIndex(doc_vectors)

user_query = "Give me contact details for Nayan Raval"
query_processor = QueryPreprocessor()
clean_query = query_processor.invoke(user_query)
query_vector = embedder.embed_query(clean_query)
query_vector = np.atleast_2d(np.array(query_vector, dtype="float32"))

D, I = faiss_index.similarity_search(query_vector, top_k=len(chunked_texts))
retrieved_chunks = [chunked_texts[i] for i in I[0] if i < len(chunked_texts)]

tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
max_model_input_tokens = 2048
max_new_tokens = 256
available_tokens_for_context = max_model_input_tokens - max_new_tokens
context_tokens = []

for chunk in retrieved_chunks:
    chunk_ids = tokenizer(chunk, return_tensors="pt")["input_ids"][0]
    if len(context_tokens) + len(chunk_ids) > available_tokens_for_context:
        remaining = available_tokens_for_context - len(context_tokens)
        if remaining > 0:
            truncated_chunk = tokenizer.decode(chunk_ids[:remaining], skip_special_tokens=True)
            context_tokens.extend(tokenizer(truncated_chunk, return_tensors="pt")["input_ids"][0])
        break
    else:
        context_tokens.extend(chunk_ids)

truncated_text = tokenizer.decode(context_tokens, skip_special_tokens=True)

reasoner = LLMReasoner(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", max_new_tokens=max_new_tokens)
answer = reasoner.invoke({"query": user_query, "context": truncated_text})

validator = AnswerValidator()
validation = validator.invoke({"answer": answer, "context": truncated_text})

# print("----------------------")
# print("User Query:", user_query)
# print("----------------------")
# print("\nAssembled Context:\n")
# print(truncated_text)

print("\n----------------------")
print("Final Answer:")
print("\n----------------------")
print(answer)
print("\nConfidence:", validation["confidence"])
print("Issues:", validation["issues"])
