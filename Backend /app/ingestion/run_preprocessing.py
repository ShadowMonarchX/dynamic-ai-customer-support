# ingestion_pipeline.py

from .data_load import DataSource
from .preprocessing import Preprocessor
from .embedding import Embedder
from .metadata_enricher import MetadataEnricher
from .ingestion_manager import IngestionManager

DATA_PATH = "/Users/jenishshekhada/Desktop/Inten/dynamic-ai-customer-support/backend/data/new_training_data.txt"


def main():
    # Step 1: Load raw documents
    try:
        source = DataSource(DATA_PATH)
        raw_documents = source.load()
        print(f"\nLoaded {len(raw_documents)} raw document(s) from: {DATA_PATH}\n")
    except FileNotFoundError as e:
        print(f"❌ Error loading data: {e}")
        return

    # Step 2: Initialize components
    preprocessor = Preprocessor(chunk_size=900, chunk_overlap=200)
    embedder = Embedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    enricher = MetadataEnricher(default_source="new_training_data")

    # Step 3: Initialize ingestion manager
    ingestion_manager = IngestionManager(
        preprocessor=preprocessor,
        embedder=embedder,
        metadata_enricher=enricher
    )

    # Step 4: Ingest documents (clean + chunk + enrich + embed)
    try:
        processed_docs, embeddings = ingestion_manager.ingest_documents(raw_documents)
    except RuntimeError as e:
        print(f"❌ Ingestion failed: {e}")
        return

    # Step 5: Summary
    print(f"\nIngestion complete! Processed {len(processed_docs)} chunks.")
    print(f"Generated embeddings shape: ({len(embeddings)}, {len(embeddings[0]) if embeddings else 0})\n")

    # Step 6: Display first few chunks and embeddings (avoid flooding console)
    display_count = min(5, len(processed_docs))
    print("========== SAMPLE CHUNKS & EMBEDDINGS ==========\n")
    for i in range(display_count):
        print(f"----- Chunk {i+1} -----")
        print(processed_docs[i].page_content[:300] + ("..." if len(processed_docs[i].page_content) > 300 else ""))
        print(f"Metadata: {processed_docs[i].metadata}")
        print(f"Embedding (first 8 dims): {embeddings[i][:8]} ...\n")


if __name__ == "__main__":
    main()
