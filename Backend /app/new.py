# import requests
# from bs4 import BeautifulSoup

# url = "https://nayanraval.vercel.app/"
# response = requests.get(url)
# soup = BeautifulSoup(response.text, "html.parser")

# text = soup.get_text()

# with open("website_data.txt", "w", encoding="utf-8") as file:
#     file.write(text)

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from typing import Dict, List, Tuple
import faiss
import re

df = pd.read_csv('/Users/jenishshekhada/Desktop/Inten/dynamic-ai-customer-support/backend /data/aa_dataset-tickets-multi-lang-5-2-50-version.csv')


print("Dataset Info:")
print(df.info())
print("\n--------------------------------\n")
print("\nFirst few rows:")
print(df.head())
print("\n--------------------------------\n")
print("\nColumn names:")
print(df.columns.tolist())
print("\n--------------------------------\n")
print("\nBasic statistics:")
print(df.describe())
print("\n--------------------------------\n")
print("\nMissing values:")
print(df.isnull().sum())


class DualEncoderRAG:
    def __init__(self, processed_df: pd.DataFrame):
        """Initialize the RAG system with dual encoder architecture."""
        self.df = processed_df

        # Check for GPU availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize embedding model
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2').to(self.device)

        # Initialize LLM for generation
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        self.generator = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small").to(self.device)

        # Initialize FAISS index
        self.setup_faiss_index()

        # Context window size
        self.max_context_tokens = 512

    def setup_faiss_index(self):
        """Create FAISS index for fast similarity search."""
        try:
            # Encode all passages
            print("Encoding passages...")
            print("\n--------------------------------\n")
            self.passages = self.df['full_query'].tolist()
            passage_embeddings = self.encoder.encode(self.passages, convert_to_tensor=True)
            passage_embeddings = passage_embeddings.cpu().numpy()  # Move to CPU for FAISS

            # Initialize FAISS index
            self.embedding_dim = passage_embeddings.shape[1]
            print(f"Embedding dimension: {self.embedding_dim}")
            print("\n--------------------------------\n")

            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.index.add(passage_embeddings.astype('float32'))
            print(f"Added {len(self.passages)} passages to the index")
            print("\n--------------------------------\n")

        except Exception as e:
            print(f"Error in setup_faiss_index: {str(e)}")
            raise


    def retrieve(self, query: str, k: int = 3) -> List[Dict]:
        """Retrieve relevant passages using the encoder."""
        try:
            # Encode query
            query_embedding = self.encoder.encode([query], convert_to_tensor=True)
            query_embedding = query_embedding.cpu().numpy()  # Move to CPU for FAISS

            # Verify dimensions
            if query_embedding.shape[1] != self.embedding_dim:
                raise ValueError(f"Query embedding dimension {query_embedding.shape[1]} does not match index dimension {self.embedding_dim}")

            # Search in FAISS index
            distances, indices = self.index.search(query_embedding.astype('float32'), k)

            # Get retrieved passages and their metadata
            retrieved = []
            for i, idx in enumerate(indices[0]):
                # Make sure we're using the actual answer field, not the subject
                retrieved.append({
                    'passage': self.passages[idx],
                    'answer': self.df.iloc[idx]['answer'],  # This should be the actual answer field
                    'score': float(distances[0][i]),
                    'metadata': {
                        'type': self.df.iloc[idx]['type'],
                        'priority': self.df.iloc[idx]['priority'],
                        'tags': self.df.iloc[idx]['tags']
                    }
                })

            return retrieved

        except Exception as e:
            print(f"Error in retrieve: {str(e)}")

            # Return a default response in case of error
            return [{
                'passage': '',
                'answer': "I apologize, but I encountered an error while processing your query.",
                'score': 0.0,
                'metadata': {
                    'type': 'error',
                    'priority': 'high',
                    'tags': ['error']
                }
            }]

    def generate(self, query: str, retrieved_contexts: List[Dict]) -> str:
        """Generate response using T5 with retrieved contexts."""
        try:
            # Filter and rank retrieved contexts
            relevant_contexts = sorted(retrieved_contexts, key=lambda x: x['score'])[:2]  # Take top 2 contexts

            # Prepare a more structured prompt
            prompt = "You are a helpful customer support assistant. Based on the following information, provide a concise and helpful response to the user's question.\n\n"

            # Add context information
            for i, ctx in enumerate(relevant_contexts):
                # Only include the answer field, not the subject or passage
                prompt += f"Reference {i+1}: {ctx['answer']}\n\n"

            # Add the user's question
            prompt += f"User Question: {query}\n\n"
            prompt += "Your Response:"

            # Tokenize and generate
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=self.max_context_tokens, truncation=True).to(self.device)

            with torch.no_grad():
                outputs = self.generator.generate(
                    inputs.input_ids,
                    max_length=150,
                    min_length=30,
                    num_beams=4,
                    temperature=0.7,
                    do_sample=True,  # Enable sampling for more natural responses
                    no_repeat_ngram_size=2
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Post-process the response
            response = self._post_process_response(response)
            return response

        except Exception as e:
            print(f"Error in generate: {str(e)}")
            return "I apologize, but I encountered an error while generating a response."

    def _post_process_response(self, response: str) -> str:
        """Post-process the generated response to remove redundant or irrelevant information."""
        # Remove any references to context numbers
        response = re.sub(r'Reference \d+:', '', response)

        # Remove any "User Question:" text that might have been included
        response = re.sub(r'User Question:.*', '', response)

        # Remove any "Your Response:" text that might have been included
        response = re.sub(r'Your Response:', '', response)

        # Remove any email-like greetings
        response = re.sub(r'Dear .*?,', '', response)
        response = re.sub(r'Hello .*?,', '', response)

        # Trim extra whitespace and newlines
        response = re.sub(r'\s+', ' ', response)
        response = response.strip()

        return response


class EnhancedCustomerSupportBot:
    def __init__(self, processed_df: pd.DataFrame):
        """Initialize the enhanced chatbot with RAG system."""
        self.rag_system = DualEncoderRAG(processed_df)
        self.conversation_history: List[Dict] = []

    def get_response(self, user_input: str) -> Dict:
        """Generate a response using RAG system."""
        # Retrieve relevant passages
        retrieved_contexts = self.rag_system.retrieve(user_input)

        # Generate response
        generated_response = self.rag_system.generate(user_input, retrieved_contexts)

        # Prepare response data
        response_data = {
            'answer': generated_response,
            'retrieved_contexts': retrieved_contexts,
            'confidence': self._calculate_confidence(retrieved_contexts),
            'metadata': self._aggregate_metadata(retrieved_contexts)
        }

        # Update conversation history
        self._update_context(user_input, response_data)

        return response_data

    def _calculate_confidence(self, retrieved_contexts: List[Dict]) -> float:
        """Calculate confidence score based on retrieved contexts."""
        if not retrieved_contexts:
            return 0.0
        # Average similarity scores of top retrieved contexts
        scores = [1.0 / (1.0 + ctx['score']) for ctx in retrieved_contexts]  # Convert distance to similarity
        return np.mean(scores)

    def _aggregate_metadata(self, retrieved_contexts: List[Dict]) -> Dict:
        """Aggregate metadata from retrieved contexts."""
        all_tags = []
        types = []
        priorities = []

        for ctx in retrieved_contexts:
            all_tags.extend(ctx['metadata']['tags'])
            types.append(ctx['metadata']['type'])
            priorities.append(ctx['metadata']['priority'])

        return {
            'tags': list(set(all_tags)),  # Remove duplicates
            'type': max(set(types), key=types.count),  # Most common type
            'priority': max(set(priorities), key=priorities.count)  # Most common priority
        }

    def _update_context(self, user_input: str, response_data: Dict) -> None:
        """Update conversation history."""
        self.conversation_history.append({
            'user_input': user_input,
            'response': response_data,
            'timestamp': pd.Timestamp.now()
        })

        # Keep only recent context
        if len(self.conversation_history) > 5:
            self.conversation_history.pop(0)


def main():
    try:
        # Load and process the dataset
        print("Loading dataset...")
        print("\n--------------------------------\n")
        df = pd.read_csv('/Users/jenishshekhada/Desktop/Inten/dynamic-ai-customer-support/backend /data/aa_dataset-tickets-multi-lang-5-2-50-version.csv')

        # Filter for English entries only
        df_en = df[df['language'] == 'en'].copy()

        # Basic preprocessing
        df_en['full_query'] = df_en['subject'].fillna('') + ' ' + df_en['body'].fillna('')

        # Make sure we're using the correct answer field
        df_en['answer'] = df_en['answer'].fillna('')

        # Clean the answer field to remove any email-like formatting
        df_en['answer'] = df_en['answer'].apply(lambda x: re.sub(r'Dear .*?,', '', x))
        df_en['answer'] = df_en['answer'].apply(lambda x: re.sub(r'Hello .*?,', '', x))
        df_en['answer'] = df_en['answer'].apply(lambda x: x.strip())

        # Keep relevant columns
        processed_df = df_en[['full_query', 'answer', 'type', 'priority', 'tag_1', 'tag_2', 'tag_3']]

        # Convert tags to list format
        processed_df['tags'] = processed_df[['tag_1', 'tag_2', 'tag_3']].values.tolist()
        processed_df = processed_df.drop(['tag_1', 'tag_2', 'tag_3'], axis=1)

        print("Initializing chatbot...")
        print("\n--------------------------------\n")
        chatbot = EnhancedCustomerSupportBot(processed_df)

        # Test queries
        test_queries = [
            "How do I reset my password?",
            "I'm having security issues with my account",
            "Can you help me with data analytics setup?",
            "What are your security compliance policies?"
        ]

        print("\nTesting Enhanced Chatbot:\n")
        print("\n--------------------------------\n")
        for query in test_queries:
            print(f"\nUser: {query}")
            response = chatbot.get_response(query)
            print(f"Bot: {response['answer']}")
            print(f"Confidence: {response['confidence']:.2f}")
            print(f"Metadata: {response['metadata']}")
            print("Top 2 Retrieved Contexts:")
            for ctx in response['retrieved_contexts'][:2]:
                print(f"- Score: {ctx['score']:.2f}")
                print(f"- Answer: {ctx['answer'][:100]}...")
            print("\n--------------------------------\n")
    except Exception as e:
        print(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()


# import face_recognition
# import numpy as np
# import requests
# from PIL import Image
# from io import BytesIO
# import time
# import json
# import os
# import traceback
# from flask import Flask, request, jsonify
# from typing import List, Dict, Any, Optional
# from waitress import serve  # pyright: ignore[reportMissingModuleSource]

# app = Flask(__name__)

# FACE_ENCODING_CACHE: Dict[str, List[np.ndarray]] = {}


# def load_image_from_url(url: str) -> Optional[np.ndarray]:
#     """Loads an image from a URL and converts it to a NumPy array."""
#     try:
#         headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
#         response = requests.get(url, headers=headers, timeout=30)
#         response.raise_for_status()

#         if "image" not in response.headers.get("Content-Type", "").lower():
#             raise ValueError("URL did not return image content.")

#         img = Image.open(BytesIO(response.content)).convert("RGB")
#         return np.array(img)
#     except requests.exceptions.RequestException as e:
#         print(f"Error (Network/Download) for {url[:50]}...: {e}")
#         return None
#     except Exception as e:
#         print(f"Error (Processing/Decoding) for {url[:50]}...: {e}")
#         return None


# def sequential_encode_all_urls(urls: List[str]) -> Dict[str, List[np.ndarray]]:
#     """
#     Sequentially loads and encodes faces from URLs, using the cache.
#     This guarantees stability by only running one resource-heavy operation at a time.
#     """
#     all_results: Dict[str, List[np.ndarray]] = {}

#     for url in urls:

#         if url in FACE_ENCODING_CACHE:
#             all_results[url] = FACE_ENCODING_CACHE[url]
#             continue

#         image = load_image_from_url(url)
#         face_encs: List[np.ndarray] = []

#         if image is not None:
#             try:

#                 face_encs = face_recognition.face_encodings(image)
#                 print(f"Processed: {url[:50]}... ({len(face_encs)} face(s))")
#             except Exception as e:
#                 print(f"Error (Encoding face_recognition) for {url[:50]}...: {e}")

#         FACE_ENCODING_CACHE[url] = face_encs
#         all_results[url] = face_encs

#     return all_results


# def is_match_in_group_photo(
#     group_encodings: list[np.ndarray],
#     known_encodings: list[np.ndarray],
#     threshold: float = 0.6,
# ) -> tuple[bool, float, int]:
#     """
#     Checks if any known face encoding matches any face in the target image.
#     Returns: (is_match: bool, best_distance: float, num_faces_found: int)
#     """

#     num_faces_found = len(group_encodings)

#     if not group_encodings:
#         return False, 1.0, 0

#     best_distance = 1.0
#     is_match = False

#     for group_face in group_encodings:
#         distances = face_recognition.face_distance(known_encodings, group_face)
#         min_dist = np.min(distances)

#         if min_dist < best_distance:
#             best_distance = min_dist

#         if min_dist <= threshold:
#             is_match = True
#             break

#     return is_match, best_distance, num_faces_found


# # flask api


# @app.route("/")
# def home():
#     return "Face recognition API is running!"


# @app.route("/match", methods=["POST"])
# def search_faces():
#     """
#     Sequential API endpoint for maximum stability.
#     Returns the simplified JSON response: match_count and a list of matching URLs.
#     """
#     start_api_time = time.time()

#     try:
#         data = request.json
#         Target_url = data.get("Target_url", [])
#         group_urls = data.get("group_urls", [])
#         # Default threshold for face matching
#         threshold = data.get("threshold", 0.55)

#         if not Target_url or not group_urls:
#             return (
#                 jsonify(
#                     {
#                         "success": False,
#                         "message": 'Both "Target_url" and "group_urls" lists must be provided and non-empty.',
#                     }
#                 ),
#                 400,
#             )

#         all_unique_urls = list(set(Target_url + group_urls))
#         print(
#             f"\n[API] Starting sequential encoding for {len(all_unique_urls)} unique URLs..."
#         )

#         encoded_results_map = sequential_encode_all_urls(all_unique_urls)

#         known_encodings = []
#         for url in Target_url:
#             known_encodings.extend(encoded_results_map.get(url, []))

#         if not known_encodings:
#             return (
#                 jsonify(
#                     {
#                         "success": False,
#                         "message": "Failed to encode any known face(s). Check input URLs.",
#                         "results": [],
#                     }
#                 ),
#                 400,
#             )

#         simplified_results: List[str] = []
#         match_count = 0

#         for url in group_urls:
#             group_encodings = encoded_results_map.get(url, [])

#             if group_encodings:
#                 is_match, _, _ = is_match_in_group_photo(
#                     group_encodings, known_encodings, threshold
#                 )

#                 if is_match:
#                     match_count += 1

#                     simplified_results.append(url)

#         end_api_time = time.time()
#         total_time = round(end_api_time - start_api_time, 2)
#         print(f"[API] Total request time: {total_time} seconds")

#         return jsonify({"match_count": match_count, "matched_urls": simplified_results})

#     except Exception as e:
#         end_api_time = time.time()
#         print("Traceback:")
#         traceback.print_exc()

#         return (
#             jsonify(
#                 {
#                     "match_count": 0,
#                     "matched_urls": [],
#                     "error": f"Internal Server Error: {type(e).__name__}",
#                     "error_time": time.ctime(),
#                 }
#             ),
#             500,
#         )


# if __name__ == "__main__":
#     try:

#         print(
#             "Flask Face Recognition API is starting. (FINAL: Sequential for STABILITY)"
#         )
#         print("Endpoint: POST /match")
#         print("Flask server is running...")

#         serve(app, host="0.0.0.0", port=5001)

#     except Exception as e:
#         print(f"\nFATAL ERROR: Could not start the server!")
#         print(f"Error detail: {e}")
