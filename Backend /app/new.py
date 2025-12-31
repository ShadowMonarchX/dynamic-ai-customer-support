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
