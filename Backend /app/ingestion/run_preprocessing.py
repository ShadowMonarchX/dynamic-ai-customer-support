from .data_load import DataSource
from .preprocessing import Preprocessor
from .embedding import Embedder  # Make sure Embedder class is correctly imported

DATA_PATH = "/Users/jenishshekhada/Desktop/Inten/dynamic-ai-customer-support/backend /data/new_training_data.txt"


def main():
    # Load raw documents
    source = DataSource(DATA_PATH)
    documents = source.load()

    print(f"\nLoaded {len(documents)} raw documents\n")

    # Preprocess documents
    pre = Preprocessor()
    cleaned_docs = pre.transform_documents(documents)

    print("\n========== CLEANED CHUNKS ==========\n")
    for i, doc in enumerate(cleaned_docs, start=1):
        print(f"----- Chunk {i} -----")
        print(doc.page_content)
        print()

    print(pre.get_stats(cleaned_docs))

    # Embed documents
    embedder = Embedder()
    embeddings = embedder.embed_documents(cleaned_docs)

    print("\n========== EMBEDDINGS ==========\n")
    for i, vec in enumerate(embeddings, start=1):
        print(f"----- Chunk {i} Vector -----")
        print(vec)  # Prints the embedding vector
        print()


if __name__ == "__main__":
    main()
