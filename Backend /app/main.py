# import os
# import numpy as np # type: ignore
# from app.ingestion.data_load import DataSource
# from app.ingestion.preprocessing import Preprocessor
# from app.ingestion.embedding import Embedded
# from app.ingestion.metadata_enricher import MetadataEnricher
# from app.ingestion.ingestion_manager import IngestionManager
# from app.vector_store.faiss_index import FAISSIndex
# from app.query_pipeline.query_preprocess import QueryPreprocessor
# from app.intent_detection.intent_classifier import IntentClassifier
# from app.intent_detection.intent_features import IntentFeaturesExtractor
# from app.reasoning.llm_reasoner import LLMReasoner
# from app.reasoning.response_generator import ResponseGenerator
# from app.validation.answer_validator import AnswerValidator
# from app.response_strategy import select_response_strategy

# data_path = '/Users/jenishshekhada/Desktop/Inten/dynamic-ai-customer-support/backend /data/training_data.txt'

# def initialize_system():
#     try:
#         if not os.path.exists(data_path):
#             raise FileNotFoundError(f"Missing knowledge base: {data_path}")

#         source = DataSource(data_path)
#         source.load_data()
        
#         processor = Preprocessor()
#         processed_docs = processor.transform_documents(source.get_documents())

#         embedder = Embedded(model_name="sentence-transformers/all-MiniLM-L6-v2")
#         vectors = embedder.embed_documents(processed_docs)
#         vectors = np.atleast_2d(np.array(vectors, dtype="float32"))

#         metadata = [doc.metadata for doc in processed_docs]
#         chunks = [doc.page_content for doc in processed_docs]
        
#         index = FAISSIndex(vectors, chunks, metadata)
        
#         return index, embedder, processor
#     except Exception as e:
#         print(f"Startup Failed: {e}")
#         exit(1)

# # Initialize Components
# faiss_index, embedder, doc_processor = initialize_system()
# classifier = IntentClassifier(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
# query_processor = QueryPreprocessor()
# generator = ResponseGenerator()  # Handles size and prompt assembly
# validator = AnswerValidator()    # Handles quality check

# print("\n--- AI Support System Ready (Jessica) ---")
# print("Type 'exit' to quit.\n")

# while True:
#     try:
#         user_input = input("Customer: ").strip()
#         if user_input.lower() in {"exit", "quit", "q"}:
#             break
#         if not user_input:
#             continue

#         # 1. Feature Engineering (Intent, Emotion, Language)
#         query_data = query_processor.invoke(user_input)
#         intent_data = classifier.classify(query_data["clean_text"])
#         features = {**query_data, **intent_data}

#         # 2. Strategy & Retrieval
#         system_strategy = select_response_strategy(features)
#         query_vector = embedder.embed_query(query_data["clean_text"])
#         query_vector = np.atleast_2d(np.array(query_vector, dtype="float32"))
        
#         retrieval_result = faiss_index.retrieve(query_vector, intent=features["intent"])
#         context_text = "\n\n".join(retrieval_result["docs"])

#         # 3. Response Generation (with Size Control)
#         # This calls your new ResponseGenerator which manages the LLMReasoner
#         answer = generator.generate({
#             "query": user_input,
#             "context": context_text,
#             "system_prompt": system_strategy,
#             "intent": features["intent"],
#             "emotion": features["emotion"],
#             "urgency": features["urgency"],
#             "complexity": features["complexity"]
#         })

#         # 4. Answer Validation (Empathy, Length, Confidence)
#         validation_result = validator.invoke({
#             "answer": answer,
#             "intent": features["intent"],
#             "emotion": features["emotion"]
#         })

#         # Output logic
#         if validation_result["valid"]:
#             print(f"AI Support Jessica: {answer}")
#         else:
#             # If validation fails, we show the answer but warn about quality
#             # Or you could trigger a "Regenerate" here.
#             print(f"AI Support Jessica (Low Confidence): {answer}")
#             # print(f"Issues: {validation_result['issues']}") # Debugging

#     except Exception as e:
#         print(f"\nSystem Error: {e}")



# import os
# import uuid
# import numpy as np

# from app.ingestion.data_load import DataSource
# from app.ingestion.preprocessing import Preprocessor
# from app.ingestion.embedding import Embedded

# from app.vector_store.faiss_index import FAISSIndex

# from app.query_pipeline.query_preprocess import QueryPreprocessor
# from app.query_pipeline.human_features import HumanFeatureExtractor
# from app.intent_detection.intent_classifier import IntentClassifier
# from app.intent_detection.intent_features import IntentFeaturesExtractor

# from app.reasoning.response_generator import ResponseGenerator
# from app.validation.answer_validator import AnswerValidator
# from app.response_strategy import ResponseStrategy


# data_path = '/Users/jenishshekhada/Desktop/Inten/dynamic-ai-customer-support/backend /data/training_data.txt'


# def initialize_system():
#     if not os.path.exists(data_path):
#         raise FileNotFoundError(f"Missing knowledge base: {data_path}")
    
#     source = DataSource(data_path)
#     source.load_data()
    
#     processor = Preprocessor()
#     processed_docs = processor.transform_documents(source.get_documents())
    
#     embedder = Embedded(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     vectors = embedder.embed_documents(processed_docs)
#     vectors = np.atleast_2d(np.array(vectors, dtype="float32"))
    
#     metadata = [doc.metadata for doc in processed_docs]
#     chunks = [doc.page_content for doc in processed_docs]
    
#     index = FAISSIndex(vectors, chunks, metadata)
#     return index, embedder


# faiss_index, embedder = initialize_system()
# query_processor = QueryPreprocessor()
# intent_classifier = IntentClassifier(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
# intent_feature_extractor = IntentFeaturesExtractor()
# generator = ResponseGenerator()
# validator = AnswerValidator()
# strategy_selector = ResponseStrategy()
# SESSION_ID = str(uuid.uuid4())

# print("\n--- AI Support System Ready (Jessica) ---")
# print("Type 'exit' to quit.\n")

# while True:
#     try:
#         user_input = input("Customer   : ").strip()
#         print("\n")
#         if user_input.lower() in {"exit", "quit", "q"}:
#             break
#         if not user_input:
#             continue

#         query_data = query_processor.invoke(user_input)

#         human_features = HumanFeatureExtractor.extract(
#             query=query_data["clean_text"],
#             session_id=SESSION_ID
#         )

#         intent_data = intent_classifier.classify(query_data["clean_text"])

#         intent_features = intent_feature_extractor.extract(
#             query=query_data["clean_text"],
#             previous_context={
#                 "intent_topic": human_features.get("previous_topic"),
#                 "question_type": human_features.get("previous_intent")
#             }
#         )

#         features = {**query_data, **intent_data, **intent_features, **human_features}

#         system_strategy = strategy_selector.select(features)

#         query_vector = embedder.embed_query(query_data["clean_text"])
#         query_vector = np.atleast_2d(np.array(query_vector, dtype="float32"))

#         retrieval = faiss_index.retrieve(
#             query_vector=query_vector,
#             intent=features.get("intent", "unknown"),
#             query_text=query_data["clean_text"]
#         )

#         context_text = "\n\n".join(retrieval.get("docs", []))

#         answer = generator.generate({
#             "query": user_input,
#             "context": context_text,
#             "system_prompt": system_strategy,
#             "intent": features.get("intent"),
#             "emotion": features.get("emotion"),
#             "urgency": features.get("urgency"),
#             "follow_up": features.get("follow_up", False)
#         })

#         validation = validator.invoke({
#             "answer": answer,
#             "intent": features.get("intent"),
#             "emotion": features.get("emotion"),
#             "similarity": 1.0  # optional: can be real similarity from FAISS
#         })

#         # print("Jessica   :", answer)
#         # print("\n")

#         if validation["confidence"] < 0.5:
#             print("Jessica: I’m not fully sure. Could you please clarify?")
#             print("\n")
#             print("issues :", validation["issues"])
#             print("confidence :", validation["confidence"])
#             print("\n")
#         else:
#             print("Jessica:", answer)
#             print("\n")
#             print("issues :", validation["issues"])
#             print("confidence :", validation["confidence"])
#             print("\n")

#     except Exception as e:
#         print(f"\nSystem Error: {e}")


import os
import uuid
import numpy as np
from app.ingestion.data_load import DataSource
from app.ingestion.preprocessing import Preprocessor
from app.ingestion.embedding import Embedded
from app.vector_store.faiss_index import FAISSIndex
from app.query_pipeline.query_preprocess import QueryPreprocessor
from app.query_pipeline.human_features import HumanFeatureExtractor
from app.intent_detection.intent_classifier import IntentClassifier
from app.intent_detection.intent_features import IntentFeaturesExtractor
from app.reasoning.response_generator import ResponseGenerator
from app.validation.answer_validator import AnswerValidator
from app.response_strategy.response_router import ResponseStrategyRouter

data_path = '/Users/jenishshekhada/Desktop/Inten/dynamic-ai-customer-support/backend /data/training_data.txt'

def initialize_system():
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
    return index, embedder

faiss_index, embedder = initialize_system()
query_processor = QueryPreprocessor()
intent_classifier = IntentClassifier(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
intent_feature_extractor = IntentFeaturesExtractor()
generator = ResponseGenerator()
validator = AnswerValidator()
strategy_router = ResponseStrategyRouter()
SESSION_ID = str(uuid.uuid4())

print("\n--- AI Support System Ready (Jessica) ---")
print("Type 'exit' to quit.\n")

while True:
    try:
        user_input = input("Customer   : ").strip()
        if user_input.lower() in {"exit", "quit", "q"}:
            break
        if not user_input:
            continue

        query_data = query_processor.invoke(user_input)
        human_features = HumanFeatureExtractor.extract(query=query_data["clean_text"], session_id=SESSION_ID)
        intent_data = intent_classifier.classify(query_data["clean_text"])
        intent_features = intent_feature_extractor.extract(
            query=query_data["clean_text"],
            previous_context={
                "intent_topic": human_features.get("previous_topic"),
                "question_type": human_features.get("previous_intent")
            }
        )

        features = {**query_data, **intent_data, **intent_features, **human_features}
        system_prompt = strategy_router.select(features)

        query_vector = embedder.embed_query(query_data["clean_text"])
        query_vector = np.atleast_2d(np.array(query_vector, dtype="float32"))

        retrieval = faiss_index.retrieve(
            query_vector=query_vector,
            intent=features.get("intent", "unknown"),
            query_text=query_data["clean_text"],
            max_chunks=5
        )

        context_text = "\n\n".join(retrieval.get("docs", []))
        answer = generator.generate({
            "query": user_input,
            "context": context_text,
            "system_prompt": system_prompt,
            "intent": features.get("intent"),
            "emotion": features.get("emotion"),
            "urgency": features.get("urgency"),
            "follow_up": features.get("follow_up", False)
        })

        validation = validator.invoke({
            "answer": answer,
            "intent": features.get("intent"),
            "emotion": features.get("emotion"),
            "similarity": retrieval.get("similarity", 1.0)
        })

        if validation["confidence"] < 0.5:
            print("Jessica: I’m not fully sure. Could you please clarify?\n")
        else:
            print("Jessica:", answer, "\n")

        print("issues :", validation["issues"])
        print("confidence :", validation["confidence"], "\n")

    except Exception as e:
        print(f"\nSystem Error: {e}")
