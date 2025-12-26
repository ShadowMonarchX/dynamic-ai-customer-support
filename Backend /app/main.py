from app.ingestion.data_load import DataSource
from app.ingestion.preprocessing import Preprocessor
from app.ingestion.embedding import Embedded
from app.vector_store.faiss_index import FAISSIndex
from app.query_pipeline.query_preprocess import QueryPreprocessor
from app.query_pipeline.query_embed import QueryEmbedder
from app.query_pipeline.context_assembler import ContextAssembler
from app.reasoning.llm_reasoner import LLMReasoner
from app.validation.answer_validator import AnswerValidator
import numpy as np # type: ignore
import os

data_path = '/Users/jenishshekhada/Desktop/Inten/dynamic-ai-customer-support/backend /data/training_data.txt'

if not os.path.exists(data_path):
    raise FileNotFoundError(data_path)

source = DataSource(data_path)
source.load_data()
documents = source.get_documents()

if not documents:
    raise ValueError("No documents loaded")

processor = Preprocessor()
processed_docs = processor.transform_documents(documents)

embedder = Embedded(model_name="sentence-transformers/all-MiniLM-L6-v2")
doc_vectors = embedder.embed_documents(processed_docs)
doc_vectors = np.array(doc_vectors, dtype="float32")

faiss_index = FAISSIndex(doc_vectors)

user_query = "Brifly explain nayan raval introduction"

query_processor = QueryPreprocessor()
clean_query = query_processor.invoke(user_query)

query_vector = embedder.embed_query(clean_query)
query_vector = np.array([query_vector], dtype="float32")

D, I = faiss_index.similarity_search(query_vector, top_k=3)
retrieved_chunks = [processed_docs[i] for i in I[0] if i < len(processed_docs)]

assembler = ContextAssembler(
    system_instructions="Answer only using the retrieved context. Do not hallucinate."
)

prompt = assembler.assemble_prompt(
    retrieved_docs=retrieved_chunks
)

context = prompt.format_prompt(
    question=user_query
).to_string()

reasoner = LLMReasoner(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
)

answer = reasoner.invoke(
    {
        "query": user_query,
        "context": context,
    }
)

validator = AnswerValidator()
validation = validator.invoke(
    {
        "answer": answer,
        "context": context,
    }
)


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