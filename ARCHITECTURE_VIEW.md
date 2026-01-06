# 🔎 System Architecture View

This document provides a **visual graph view** of the Dynamic AI Customer Support backend.
The diagram represents both **Offline (Ingestion)** and **Online (Query)** pipelines.

---

## 🧠 High-Level Architecture Graph

```mermaid
flowchart LR

%% ======================
%% Offline Pipeline
%% ======================
subgraph OFFLINE["Offline Timeline (Ingestion)"]
    A[Raw Training Data\ntraining_data.txt]
    B[Data Loader]
    C[Text Preprocessing\nCleaning & Chunking]
    D[Metadata Enrichment]
    E[Embedding Generator\nSentence Transformers]
    F[FAISS Vector Index]

    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
end

%% ======================
%% Online Pipeline
%% ======================
subgraph ONLINE["Online Timeline (User Query)"]
    U[User Query]
    QP[Query Preprocessor]
    HF[Human Feature Extractor]
    IC[Intent & Emotion Classifier]
    QS[Response Strategy Router]

    QE[Query Embedder]
    RR[Retrieval Router]
    CA[Context Assembler]

    LLM[LLM Reasoner\nResponse Generator]
    VAL[Answer Validator]
    RESP[Final Response]

    U --> QP
    QP --> HF
    QP --> IC
    HF --> QS
    IC --> QS

    QP --> QE
    QE --> RR
    RR --> CA

    CA --> LLM
    QS --> LLM
    LLM --> VAL
    VAL --> RESP
end

%% ======================
%% Cross-Pipeline Links
%% ======================
F --> RR
