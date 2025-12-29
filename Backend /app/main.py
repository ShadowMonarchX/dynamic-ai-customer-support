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
import re
import numpy as np  # type: ignore

from transformers import AutoTokenizer  # type: ignore
from langchain_core.documents import Document  # type: ignore

from app.ingestion.data_load import DataSource
from app.ingestion.preprocessing import Preprocessor
from app.ingestion.embedding import Embedded
from app.vector_store.faiss_index import FAISSIndex
from app.query_pipeline.query_preprocess import QueryPreprocessor
from app.intent_detection.intent_classifier import IntentClassifier
from app.reasoning.llm_reasoner import LLMReasoner
from app.validation.answer_validator import AnswerValidator
from app.response_strategy import (
    GreetingResponseStrategy,
    FAQResponseStrategy,
    TransactionalResponseStrategy,
    EmotionResponseStrategy,
    BigIssueResponseStrategy,
)


data_path = '/Users/jenishshekhada/Desktop/Inten/dynamic-ai-customer-support/backend /data/training_data.txt'

if not os.path.exists(data_path):
    raise FileNotFoundError(f"Data file not found: {data_path}")

source = DataSource(data_path)
source.load_data()
documents = source.get_documents()

if not documents:
    raise ValueError("No documents loaded")

processor = Preprocessor()
processed_docs = processor.transform_documents(documents)

if not processed_docs:
    raise ValueError("No processed documents")



def split_text(text: str, tokenizer, max_tokens: int = 500):
    sentences = re.split(r"(?<=[.!?]) +", text)
    chunks, current, tokens = [], "", 0

    for s in sentences:
        s_tokens = len(tokenizer(s, return_tensors="pt")["input_ids"][0])
        if tokens + s_tokens > max_tokens:
            chunks.append(current.strip())
            current, tokens = s, s_tokens
        else:
            current += " " + s
            tokens += s_tokens

    if current:
        chunks.append(current.strip())
    return chunks


chunk_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

chunks = []
for doc in processed_docs:
    text = doc.page_content if isinstance(doc, Document) else str(doc)
    chunks.extend(split_text(text, chunk_tokenizer))


embedder = Embedded(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectors = embedder.embed_documents([Document(page_content=c) for c in chunks])
vectors = np.atleast_2d(np.array(vectors, dtype="float32"))

faiss_index = FAISSIndex(vectors)



user_query = input("Enter your question: ").strip()

query_processor = QueryPreprocessor()
query_data = query_processor.invoke(user_query)

clean_text = query_data["clean_text"]
urgency = query_data["urgency"]
emotion = query_data["emotion"]

classifier = IntentClassifier(model="llama3")
intent_data = classifier.classify(clean_text)

intent = intent_data["intent"]


if intent == "greeting":
    answer = GreetingResponseStrategy().generate_response()

elif intent == "faq":
    answer = FAQResponseStrategy().generate_response(
        "This service provides backend development support."
    )

elif intent == "transactional":
    answer = TransactionalResponseStrategy().generate_response(
        "I can help with refunds, cancellations, or order tracking."
    )

elif intent == "account_support":
    answer = EmotionResponseStrategy().generate_response(emotion)

elif intent == "big_issue":
    print(BigIssueResponseStrategy().generate_response())

else:
    
    query_vector = embedder.embed_query(clean_text)
    query_vector = np.atleast_2d(np.array(query_vector, dtype="float32"))

    top_k = 4 if intent == "big_issue" else 1
    _, indices = faiss_index.similarity_search(query_vector, top_k=top_k)

    context_chunks = [chunks[i] for i in indices[0] if i < len(chunks)]
    context = "\n".join(context_chunks)

    reasoner = LLMReasoner()
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
        "intent": intent,
        "emotion": emotion,
    }
)



print("\n----------------------")
print("Final Answer:")
print("----------------------")

print(validation["answer"])
print("\nConfidence:", validation["confidence"])
print("Issues:", validation["issues"])
