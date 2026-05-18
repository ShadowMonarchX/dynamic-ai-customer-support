# ingestion_pipeline.py

from .data_load import DataSource
from .preprocessing import Preprocessor
from .embedding import Embedder
from .metadata_enricher import MetadataEnricher
from .ingestion_manager import IngestionManager

DATA_PATH = "/Users/jenishshekhada/Desktop/Inten/dynamic-ai-customer-support/backend /data/training_data.txt"


def main():
    try:
        source = DataSource(DATA_PATH)
        raw_documents = source.load()
        print(f"\nLoaded {len(raw_documents)} raw document(s) from: {DATA_PATH}\n")
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return

    preprocessor = Preprocessor(chunk_size=900, chunk_overlap=200)
    embedder = Embedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    enricher = MetadataEnricher(default_source="training_data")

    ingestion_manager = IngestionManager(
        preprocessor=preprocessor, embedder=embedder, metadata_enricher=enricher
    )

    try:
        processed_docs, embeddings = ingestion_manager.ingest_documents(raw_documents)
    except RuntimeError as e:
        print(f"Ingestion failed: {e}")
        return

    print(f"\nIngestion complete! Total chunks: {len(processed_docs)}\n")

    for i, doc in enumerate(processed_docs, start=1):
        print(f"----- Chunk {i} -----")
        print(doc.page_content)
        print(f"Metadata: {doc.metadata}")
        print(f"Embedding (first 8 dims): {embeddings[i-1][:8]} ...\n")


if __name__ == "__main__":
    main()
