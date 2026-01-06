# 🔎 System Architecture View

This document provides a **visual graph view** of the Dynamic AI Customer Support backend.
The diagram represents both **Offline (Ingestion)** and **Online (Query)** pipelines.

---

## 🧠 High-Level Architecture Graph

```mermaid
flowchart LR

%% ===============================
%% DATA LAYER
%% ===============================
subgraph DATA["Data Layer"]
    D1["Raw Knowledge Sources<br/>TXT | Docs | FAQs"]
end

%% ===============================
%% OFFLINE INGESTION PIPELINE
%% ===============================
subgraph OFFLINE["Offline Timeline - Knowledge Ingestion"]
    O1["Data Loader"]
    O2["Text Preprocessing<br/>Cleaning | Normalization | Headers"]
    O3["Chunking Engine<br/>Fixed Size | Overlap"]
    O4["Metadata Enricher<br/>Source | Section | Topic"]
    O5["Embedding Generator<br/>Sentence Transformers"]
    O6["Vector Quality Checks<br/>Shape | Consistency"]
    O7["FAISS Vector Index<br/>Vectors + Chunks + Metadata"]

    D1 --> O1
    O1 --> O2
    O2 --> O3
    O3 --> O4
    O4 --> O5
    O5 --> O6
    O6 --> O7
end

%% ===============================
%% API AND SESSION LAYER
%% ===============================
subgraph API["API and Session Layer"]
    API1["FastAPI Server"]
    API2["Session Manager<br/>UUID | Context"]
end

%% ===============================
%% ONLINE QUERY PIPELINE
%% ===============================
subgraph ONLINE["Online Timeline - User Interaction"]
    U["User Query"]

    Q1["Query Preprocessor"]
    Q2["Human Feature Extractor<br/>Urgency | History"]
    Q3["Intent Classifier<br/>Greeting | Question | Complaint"]
    Q4["Emotion Analyzer<br/>Angry | Neutral | Calm"]

    S1["Response Strategy Router<br/>Tone Selection"]

    Q5["Query Embedder"]
    R1["Retrieval Router"]
    R2["Vector Search<br/>FAISS kNN"]
    R3["Context Assembler<br/>Ranked Chunks"]

    L1["LLM Reasoning Engine<br/>Prompt + Context"]
    V1["Answer Validator<br/>Grounding | Confidence"]
    OUT["Final Response"]

    U --> API1
    API1 --> API2
    API2 --> Q1

    Q1 --> Q2
    Q1 --> Q3
    Q3 --> Q4

    Q2 --> S1
    Q3 --> S1
    Q4 --> S1

    Q1 --> Q5
    Q5 --> R1
    R1 --> R2
    R2 --> R3

    R3 --> L1
    S1 --> L1

    L1 --> V1
    V1 --> OUT
end

%% ===============================
%% CROSS PIPELINE LINK
%% ===============================
O7 --> R2
