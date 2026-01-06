# 🧠 Dynamic AI Customer Support Backend

An end-to-end **AI-powered customer support backend** built with **FastAPI**, combining **offline ingestion**, **vector search (FAISS)**, **intent detection**, **human feature extraction**, **LLM reasoning**, and **response strategy selection**.

This system is designed using **clean modular architecture**, separating **offline processing** and **online query execution** for scalability and maintainability.

---

## 🚀 Key Features

* 🔹 Offline document ingestion & preprocessing
* 🔹 Chunking + embeddings using Sentence Transformers
* 🔹 FAISS-based vector similarity search
* 🔹 Intent & emotion detection
* 🔹 Human behavior feature extraction
* 🔹 Context-aware retrieval routing
* 🔹 LLM-powered response generation
* 🔹 Answer validation to reduce hallucinations
* 🔹 Strategy-based response selection
* 🔹 FastAPI REST interface

---

## 🏗️ System Architecture Overview

### Offline Timeline (One-Time / Batch Process)

1. Load raw text data
2. Clean & preprocess documents
3. Chunk large documents
4. Generate embeddings
5. Enrich metadata
6. Store vectors in FAISS index

### Online Timeline (Per User Query)

1. Preprocess user query
2. Extract human behavior features
3. Detect intent & emotion
4. Route retrieval strategy
5. Retrieve relevant chunks
6. Assemble contextual prompt
7. Generate response using LLM
8. Validate answer confidence
9. Return final response

---

## 📁 Project Structure

```text
backend/
├── app/
│   ├── main.py                # Application entry point
│   ├── __init__.py
│
│   ├── ingestion/             # Offline ingestion pipeline
│   │   ├── data_load.py
│   │   ├── preprocessing.py
│   │   ├── embedding.py
│   │   ├── metadata_enricher.py
│   │   ├── ingestion_manager.py
│   │   ├── run_preprocessing.py
│   │   └── __init__.py
│
│   ├── intent_detection/      # Intent & emotion detection
│   │   ├── intent_classifier.py
│   │   ├── intent_features.py
│   │   └── __init__.py
│
│   ├── query_pipeline/        # Online query processing
│   │   ├── query_preprocess.py
│   │   ├── human_features.py
│   │   ├── query_embed.py
│   │   ├── context_assembler.py
│   │   ├── retrieval_router.py
│   │   └── __init__.py
│
│   ├── vector_store/           # Vector storage layer
│   │   ├── faiss_index.py
│   │   └── __init__.py
│
│   ├── reasoning/              # LLM reasoning
│   │   ├── llm_reasoner.py
│   │   ├── response_generator.py
│   │   └── __init__.py
│
│   ├── response_strategy/      # Response style selection
│   │   ├── response_router.py
│   │   ├── response_strategy.py
│   │   └── __init__.py
│
│   ├── validation/             # Answer validation
│   │   ├── answer_validator.py
│   │   └── __init__.py
│
│   └── data/
│       └── training_data.txt   # Knowledge base
```

---

## 🧩 Core Components Explained

### 🔹 Ingestion Pipeline (`ingestion/`)

Handles offline data preparation:

* Reads large text files
* Cleans & chunks content
* Generates embeddings
* Enriches metadata
* Prepares data for vector storage

### 🔹 Intent Detection (`intent_detection/`)

Detects:

* User intent (greeting, question, complaint, etc.)
* Emotional tone (angry, neutral, urgent)

### 🔹 Query Pipeline (`query_pipeline/`)

Online query execution:

* Cleans user input
* Extracts human behavioral features
* Embeds queries
* Retrieves relevant context

### 🔹 Vector Store (`vector_store/`)

* FAISS-based similarity search
* Efficient nearest-neighbor lookup

### 🔹 Reasoning Engine (`reasoning/`)

* Uses LLM to generate answers from retrieved context
* Applies system prompts dynamically

### 🔹 Response Strategy (`response_strategy/`)

* Chooses response tone (polite, empathetic, concise, etc.)
* Adjusts based on intent & emotion

### 🔹 Validation (`validation/`)

* Ensures answers are grounded in context
* Reduces hallucinations via confidence scoring

---

## ⚙️ Tech Stack

* **Backend Framework:** FastAPI
* **Embeddings:** Sentence Transformers (`all-MiniLM-L6-v2`)
* **Vector Search:** FAISS
* **LLM:** TinyLlama 1.1B Chat
* **Data Processing:** NumPy
* **API Schema:** Pydantic

---

## ▶️ Running the Application

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Set Training Data Path

Update in `app/main.py`:

```python
DATA_PATH = "/path/to/training_data.txt"
```

### 3️⃣ Start the Server

```bash
uvicorn app.main:app --reload
```

### 4️⃣ API Endpoints

* **Health Check**

```http
GET /
```

* **Query Chatbot**

```http
POST /query
Content-Type: application/json

{
  "user_query": "How do I reset my password?"
}
```

---

## 🧪 Example Response Flow

1. User sends a query
2. Intent + emotion detected
3. Context retrieved from FAISS
4. LLM generates response
5. Validator checks confidence
6. Final answer returned

---

## 📌 Future Improvements

* Streaming responses
* Multi-language support
* Persistent session memory
* Redis-based caching
* Async ingestion
* Hybrid search (BM25 + vectors)

---

## 👨‍💻 Author

**Jenish Shekhada**
AI Engineer | GenAI | RAG Systems | FastAPI


