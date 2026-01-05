import os
import uuid
import logging
import numpy as np

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.data_ingestion.data_load import DataSource
from app.data_ingestion.preprocessing import Preprocessor
from app.data_ingestion.embedding import Embedder
from app.data_ingestion.metadata_enricher import MetadataEnricher
from app.data_ingestion.ingestion_manager import IngestionManager

from app.vector_store.faiss_index import FAISSIndex

from app.query_pipeline.query_preprocess import QueryPreprocessor
from app.query_pipeline.human_features import HumanFeatureExtractor
from app.query_pipeline.query_embed import QueryEmbedder
from app.query_pipeline.context_assembler import ContextAssembler
from app.query_pipeline.retrieval_router import RetrievalRouter

from app.intent_detection.intent_classifier import IntentClassifier
from app.intent_detection.intent_features import IntentFeaturesExtractor

from app.reasoning.response_generator import ResponseGenerator

from app.validation.answer_validator import AnswerValidator

from app.response_strategy.response_router import ResponseStrategyRouter

DATA_PATH = "/Users/jenishshekhada/Desktop/Inten/dynamic-ai-customer-support/backend/app/data/training_data.txt"

app = FastAPI(title="AI Chatbot API")
SESSION_ID = str(uuid.uuid4())


class QueryRequest(BaseModel):
    user_query: str


def initialize_system():
    try:
        source = DataSource(DATA_PATH)
        raw_documents = source.load()

        preprocessor = Preprocessor(chunk_size=900, chunk_overlap=200)
        embedder = Embedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
        enricher = MetadataEnricher(default_source="training_data")

        ingestion_manager = IngestionManager(
            preprocessor=preprocessor,
            embedder=embedder,
            metadata_enricher=enricher,
        )

        processed_docs, embeddings = ingestion_manager.ingest_documents(raw_documents)

        vectors = np.atleast_2d(np.array(embeddings, dtype="float32"))
        metadata = [doc.metadata for doc in processed_docs]
        chunks = [doc.page_content for doc in processed_docs]

        index = FAISSIndex(vectors, chunks, metadata)
        return index, embedder
    except Exception as e:
        raise RuntimeError(f"Initialization failed: {e}")


# Initialize system on startup
try:
    print("Initializing AI Support System...")
    faiss_index, embedder = initialize_system()
    print("Initialization Complete.")

    query_processor = QueryPreprocessor()
    query_embedder = QueryEmbedder(embedder)
    context_assembler = ContextAssembler()
    retriever = RetrievalRouter(query_embedder, faiss_index)

    intent_classifier = IntentClassifier(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )
    intent_feature_extractor = IntentFeaturesExtractor()

    generator = ResponseGenerator()
    validator = AnswerValidator()
    strategy_router = ResponseStrategyRouter()

    print("AI Support System Ready.")

except Exception as e:
    print(f"Fatal Initialization Error: {e}")
    raise e


@app.get("/")
def root():
    return {"message": "AI Chatbot API is running"}


@app.post("/query")
def query_chatbot(request: QueryRequest):
    try:
        user_input = request.user_query.strip()
        if not user_input:
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        query_data = query_processor.invoke(user_input)

        human_features = HumanFeatureExtractor.extract(
            query=query_data["clean_text"],
            session_id=SESSION_ID,
        )

        intent_data = intent_classifier.classify(query_data["clean_text"])

        if intent_data.get("intent") == "greeting":
            return {"response": "Hi! How can I help you today?", "intent": "greeting"}

        intent_features = intent_feature_extractor.extract(
            query=query_data["clean_text"],
            previous_context={
                "intent_topic": human_features.get("previous_topic"),
                "question_type": human_features.get("previous_intent"),
            },
        )

        features = {
            **query_data,
            **intent_data,
            **intent_features,
            **human_features,
        }

        system_prompt = strategy_router.select(features)

        retrieval = retriever.retrieve(
            query=query_data["clean_text"],
            top_k=5,
        )

        if not retrieval:
            return {"response": "I’m not fully sure. Could you please clarify?"}

        context_text = context_assembler.assemble(
            retrieval=retrieval,
            intent=features.get("intent"),
        )

        answer = generator.generate(
            {
                "query": user_input,
                "context": context_text,
                "system_prompt": system_prompt,
                "intent": features.get("intent"),
                "emotion": features.get("emotion"),
                "urgency": features.get("urgency"),
                "follow_up": features.get("follow_up", False),
            }
        )

        validation = validator.invoke(
            {
                "answer": answer,
                "intent": features.get("intent"),
                "emotion": features.get("emotion"),
                "similarity": 1.0,
                "context": context_text,
            }
        )

        if validation["confidence"] < 0.5:
            return {
                "response": "I’m not fully sure. Could you please clarify?",
                "confidence": validation["confidence"],
            }
        else:
            return {"response": answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# def initialize_system():
#     try:
#         source = DataSource(DATA_PATH)
#         raw_documents = source.load()

#         preprocessor = Preprocessor(chunk_size=900, chunk_overlap=200)
#         embedder = Embedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
#         enricher = MetadataEnricher(default_source="training_data")

#         ingestion_manager = IngestionManager(
#             preprocessor=preprocessor,
#             embedder=embedder,
#             metadata_enricher=enricher,
#         )

#         processed_docs, embeddings = ingestion_manager.ingest_documents(raw_documents)

#         vectors = np.atleast_2d(np.array(embeddings, dtype="float32"))
#         metadata = [doc.metadata for doc in processed_docs]
#         chunks = [doc.page_content for doc in processed_docs]

#         index = FAISSIndex(vectors, chunks, metadata)
#         return index, embedder
#     except Exception as e:
#         raise RuntimeError(f"Initialization failed: {e}")


# try:
#     print("\nInitializing AI Support System...\n")
#     faiss_index, embedder = initialize_system()
#     print("\nInitialization Complete.\n")

#     query_processor = QueryPreprocessor()
#     query_embedder = QueryEmbedder(embedder)
#     context_assembler = ContextAssembler()
#     retriever = RetrievalRouter(query_embedder, faiss_index)

#     intent_classifier = IntentClassifier(
#         model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
#     )
#     intent_feature_extractor = IntentFeaturesExtractor()

#     generator = ResponseGenerator()
#     validator = AnswerValidator()
#     strategy_router = ResponseStrategyRouter()

#     SESSION_ID = str(uuid.uuid4())

#     print("\nAI Support System Ready\n")

#     while True:
#         try:
#             user_input = input("\nCustomer  : ").strip()
#             if user_input.lower() in {"exit", "quit", "q"}:
#                 break
#             if not user_input:
#                 continue

#             query_data = query_processor.invoke(user_input)

#             human_features = HumanFeatureExtractor.extract(
#                 query=query_data["clean_text"],
#                 session_id=SESSION_ID,
#             )

#             intent_data = intent_classifier.classify(query_data["clean_text"])

#             if intent_data.get("intent") == "greeting":
#                 print("Jessica  :  Hi! How can I help you today?\n")
#                 continue

#             intent_features = intent_feature_extractor.extract(
#                 query=query_data["clean_text"],
#                 previous_context={
#                     "intent_topic": human_features.get("previous_topic"),
#                     "question_type": human_features.get("previous_intent"),
#                 },
#             )

#             features = {
#                 **query_data,
#                 **intent_data,
#                 **intent_features,
#                 **human_features,
#             }

#             system_prompt = strategy_router.select(features)

#             retrieval = retriever.retrieve(
#                 query=query_data["clean_text"],
#                 top_k=5,
#             )

#             if not retrieval:
#                 print("Jessica  :  I’m not fully sure. Could you please clarify?\n")
#                 continue

#             context_text = context_assembler.assemble(
#                 retrieval=retrieval,
#                 intent=features.get("intent"),
#             )

#             answer = generator.generate(
#                 {
#                     "query": user_input,
#                     "context": context_text,
#                     "system_prompt": system_prompt,
#                     "intent": features.get("intent"),
#                     "emotion": features.get("emotion"),
#                     "urgency": features.get("urgency"),
#                     "follow_up": features.get("follow_up", False),
#                 }
#             )

#             # validation = validator.invoke(
#             #     {
#             #         "answer": answer,
#             #         "intent": features.get("intent"),
#             #         "emotion": features.get("emotion"),
#             #         "similarity": 1.0,
#             #     }
#             # )
#             validation = validator.invoke(
#                 {
#                     "answer": answer,
#                     "intent": features.get("intent"),
#                     "emotion": features.get("emotion"),
#                     "similarity": 1.0,
#                     "context": context_text,  # Add this line
#                 }
#             )

#             # print("\n")
#             # print("validation", validation)
#             # print("--------------------")
#             # print("\n---- Validation Scores ----")
#             # print("\nconfidence  : ", validation["confidence"])
#             # print("relevance  : ", validation["relevance"])
#             # print("clarity    : ", validation["clarity"])
#             # print("consistency: ", validation["consistency"])
#             # print("completeness: ", validation["completeness"])
#             # print("final_score: ", validation["final_score"], "\n")
#             # print("---- Debug Info ----")
#             # print("Intent     : ", features.get("intent"))
#             # print("Emotion    : ", features.get("emotion"))
#             # print("Urgency    : ", features.get("urgency"))
#             # print("Follow Up  : ", features.get("follow_up"))
#             # print("--------------------\n")
#             if validation["confidence"] < 0.5:
#                 print("Jessica  : I’m not fully sure. Could you please clarify?\n")
#             else:
#                 print(f"Jessica  : {answer}\n")

#             print(f"confidence  : {validation['confidence']}")

#         except Exception as e:
#             print(f"Runtime Error  : {e}")

# except Exception as e:
#     print(f"Fatal Error  : {e}")
