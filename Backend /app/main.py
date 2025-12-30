import os
import numpy as np # type: ignore
import threading
from transformers import AutoTokenizer # type: ignore
from langchain_core.documents import Document # type: ignore

from app.ingestion.data_load import DataSource
from app.ingestion.preprocessing import Preprocessor
from app.ingestion.embedding import Embedded
from app.vector_store.faiss_index import FAISSIndex
from app.query_pipeline.query_preprocess import QueryPreprocessor
from app.intent_detection.intent_classifier import IntentClassifier
from app.reasoning.llm_reasoner import LLMReasoner
from app.response_strategy import (
    select_response_strategy,  # Add this line
    GreetingResponse,
    FAQResponse,
    TransactionalResponse,
    EmotionResponse,
    BigIssueResponse,
)
data_path = '/Users/jenishshekhada/Desktop/Inten/dynamic-ai-customer-support/backend /data/training_data.txt'

def initialize_system():
    try:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Missing knowledge base: {data_path}")

        source = DataSource(data_path)
        source.load_data()
        
        processor = Preprocessor()
        processed_docs = processor.transform_documents(source.get_documents())

        embedder = Embedded(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectors = embedder.embed_documents(processed_docs)
        vectors = np.atleast_2d(np.array(vectors, dtype="float32"))

        metadata = [doc.metadata for doc in processed_docs]
        chunks = [doc.page_content for doc in processed_docs]
        
        index = FAISSIndex(vectors, chunks, metadata)
        
        return index, embedder, processor
    except Exception as e:
        print(f"Startup Failed: {e}")
        exit(1)

faiss_index, embedder, doc_processor = initialize_system()
classifier = IntentClassifier(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
query_processor = QueryPreprocessor()
reasoner = LLMReasoner()

print("AI Support System Ready. Type 'exit' to quit.")

while True:
    try:
        user_input = input("\nCustomer: ").strip()
        if user_input.lower() in {"exit", "quit", "q"}:
            break
        if not user_input:
            continue

        # Step 1: Online Feature Engineering
        query_data = query_processor.invoke(user_input)
        intent_data = classifier.classify(query_data["clean_text"])
        
        # Merge all features
        features = {**query_data, **intent_data}

        # Step 2: Strategy Selection
        system_strategy = select_response_strategy(features)

        # Step 3: Retrieval (Filtered by Intent)
        query_vector = embedder.embed_query(query_data["clean_text"])
        query_vector = np.atleast_2d(np.array(query_vector, dtype="float32"))
        
        retrieval_result = faiss_index.retrieve(query_vector, intent=features["intent"])
        context_text = "\n\n".join(retrieval_result["docs"])

        # Step 4: Reasoning & Generation
        answer = reasoner.invoke({
            "query": user_input,
            "context": context_text,
            "system_prompt": system_strategy,
            "intent": features["intent"],
            "emotion": features["emotion"],
            "urgency": features["urgency"],
            "complexity": features["complexity"]
        })

        # print(f"\nAI Support [{features['intent']}]: {answer}")
        print(f"\nAI Support Jessica : {answer}")

    except Exception as e:
        print(f"\nSystem Error: {e}")