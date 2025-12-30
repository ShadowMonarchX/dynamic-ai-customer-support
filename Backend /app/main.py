# import os
# import numpy as np #type: ignore
# from app.ingestion.data_load import DataSource
# from app.ingestion.preprocessing import Preprocessor
# from app.ingestion.embedding import Embedded
# from app.vector_store.faiss_index import FAISSIndex
# from app.query_pipeline.query_preprocess import QueryPreprocessor
# from app.reasoning.llm_reasoner import LLMReasoner
# from app.validation.answer_validator import AnswerValidator
# from app.intent_detection.intent_classifier import IntentClassifier
# from app.response_strategy import (
#     GreetingResponseStrategy,
#     FAQResponseStrategy,
#     TransactionalResponseStrategy,
#     EmotionResponseStrategy,
#     BigIssueResponseStrategy,
# )
# from langchain_core.documents import Document #type: ignore
# from transformers import AutoTokenizer #type: ignore
# import re


# data_path = '/Users/jenishshekhada/Desktop/Inten/dynamic-ai-customer-support/backend /data/training_data.txt'


# if not os.path.exists(data_path):
#     raise FileNotFoundError(f"Data file not found: {data_path}")


# source = DataSource(data_path)
# source.load_data()
# documents = source.get_documents()


# if not documents:
#     raise ValueError("No documents loaded from the file.")


# processor = Preprocessor()
# processed_docs = processor.transform_documents(documents)


# if not processed_docs:
#     raise ValueError("No processed documents available.")


# def split_text_into_chunks(text, tokenizer, max_tokens=500):
#     sentences = re.split(r'(?<=[.!?]) +', text)
#     chunks = []
#     current_chunk = ""
#     current_tokens = 0
#     for sent in sentences:
#         sent_tokens = len(tokenizer(sent, return_tensors="pt")["input_ids"][0])
#         if current_tokens + sent_tokens > max_tokens:
#             if current_chunk:
#                 chunks.append(current_chunk.strip())
#             current_chunk = sent
#             current_tokens = sent_tokens
#         else:
#             current_chunk += " " + sent
#             current_tokens += sent_tokens
#     if current_chunk:
#         chunks.append(current_chunk.strip())
#     return chunks


# tokenizer_for_chunks = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
# chunked_texts = []
# for doc in processed_docs:
#     text = doc.page_content if isinstance(doc, Document) else str(doc)
#     chunked_texts.extend(split_text_into_chunks(text, tokenizer_for_chunks, max_tokens=500))


# embedder = Embedded(model_name="sentence-transformers/all-MiniLM-L6-v2")
# doc_vectors = embedder.embed_documents([Document(page_content=chunk) for chunk in chunked_texts])
# doc_vectors = np.atleast_2d(np.array(doc_vectors, dtype="float32"))
# faiss_index = FAISSIndex(doc_vectors)


# # user_query = "what is backend development services "
# # user_query = input("Enter your Question  : ")

# # query_classifier = IntentClassifier(model="llama3")
# # intent_result = query_classifier.classify(user_query)

# # query_processor = QueryPreprocessor()
# # clean_query = query_processor.invoke(user_query,intent=intent_result.get("intent", "unknown"))




# user_query = input("Enter your Question: ").strip()

# query_processor = QueryPreprocessor()
# query_data = query_processor.invoke(user_query)

# clean_text = query_data["clean_text"]
# urgency = query_data["urgency"]
# emotion = query_data["emotion"]


# classifier = IntentClassifier(model="llama3")
# intent_result = classifier.classify(clean_text)

# intent = intent_result["intent"]


# if intent == "greeting":
#     print(GreetingResponseStrategy().generate_response())

# elif intent == "faq":
#     print(
#         FAQResponseStrategy().generate_response(
#             "This service provides backend development support."
#         )
#     )

# elif intent == "transactional":
#     print(
#         TransactionalResponseStrategy().generate_response(
#             "I can help with refunds, cancellations, or order tracking."
#         )
#     )

# elif intent == "big_issue":
#     print(BigIssueResponseStrategy().generate_response())

# elif intent == "account_support":
#     print(EmotionResponseStrategy().generate_response(emotion))

# else:
#     print("Could you please provide a bit more detail so I can assist you?")


# query_vector = embedder.embed_query(clean_query)
# query_vector = np.atleast_2d(np.array(query_vector, dtype="float32"))

# D, I = faiss_index.similarity_search(query_vector, top_k=len(chunked_texts))
# retrieved_chunks = [chunked_texts[i] for i in I[0] if i < len(chunked_texts)]


# tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
# max_model_input_tokens = 2048
# max_new_tokens = 256
# available_tokens_for_context = max_model_input_tokens - max_new_tokens
# context_tokens = []


# for chunk in retrieved_chunks:
#     chunk_ids = tokenizer(chunk, return_tensors="pt")["input_ids"][0]
#     if len(context_tokens) + len(chunk_ids) > available_tokens_for_context:
#         remaining = available_tokens_for_context - len(context_tokens)
#         if remaining > 0:
#             truncated_chunk = tokenizer.decode(chunk_ids[:remaining], skip_special_tokens=True)
#             context_tokens.extend(tokenizer(truncated_chunk, return_tensors="pt")["input_ids"][0])
#         break
#     else:
#         context_tokens.extend(chunk_ids)


# truncated_text = tokenizer.decode(context_tokens, skip_special_tokens=True)


# reasoner = LLMReasoner(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", max_new_tokens=max_new_tokens)
# answer = reasoner.invoke({"query": user_query, "context": truncated_text})


# validator = AnswerValidator()
# validation = validator.invoke({"answer": answer, "context": truncated_text})


# # print("----------------------")
# # print("User Query:", user_query)
# # print("----------------------")
# # print("\nAssembled Context:\n")
# # print(truncated_text)

# print("\n----------------------")
# print("Final Answer:")
# print("\n----------------------")
# print(answer)
# print("\nConfidence:", validation["confidence"])
# print("Issues:", validation["issues"])

import os
import numpy as np
import threading
from transformers import AutoTokenizer
from langchain_core.documents import Document

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

        print(f"\nAI Support [{features['intent']}]: {answer}")

    except Exception as e:
        print(f"\nSystem Error: {e}")