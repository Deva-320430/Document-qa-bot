"""
Indexing entry point.

Run this script once (or whenever documents in data/ change) to:
  1. Load all documents from data/
  2. Split them into overlapping chunks
  3. Embed the chunks and store them in a persistent ChromaDB vector store

Usage:
    python main.py
"""

from src.ingest import load_documents
from src.utils import chunk_documents
from src.rag import create_vector_store


def main():
    print("Step 1/3: Loading documents from data/ ...")
    documents = load_documents("data")
    print(f"  -> Loaded {len(documents)} document section(s) total.\n")

    print("Step 2/3: Splitting documents into chunks ...")
    chunks = chunk_documents(documents)
    print(f"  -> Created {len(chunks)} chunks (chunk_size=1000, overlap=200).\n")

    print("Step 3/3: Embedding chunks and building the vector store ...")
    print("  (first run will download the embedding model, this may take a minute)")
    create_vector_store(chunks)

    print(f"\nIndexing complete: {len(chunks)} chunks indexed from {len(documents)} document section(s).")
    print("You can now run 'python query_bot.py' to ask questions.")


if __name__ == "__main__":
    main()
