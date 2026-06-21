"""
Document ingestion module.
Loads all supported documents (PDF and TXT) from the data directory.
"""

import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader


def load_documents(data_path="data"):
    """
    Load all .pdf and .txt files from the given directory.

    Each loaded document carries metadata (e.g. 'source', and 'page' for PDFs)
    that is preserved through chunking and used later for citations.

    Returns:
        list[Document]: LangChain Document objects with page_content + metadata.
    """
    documents = []

    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    for file in sorted(os.listdir(data_path)):
        path = os.path.join(data_path, file)

        if not os.path.isfile(path):
            continue

        if file.lower().endswith(".pdf"):
            loader = PyPDFLoader(path)
            loaded = loader.load()
            documents.extend(loaded)
            print(f"  Loaded {len(loaded)} page(s) from {file}")

        elif file.lower().endswith(".txt"):
            loader = TextLoader(path, encoding="utf-8")
            loaded = loader.load()
            documents.extend(loaded)
            print(f"  Loaded {file}")

        else:
            print(f"  Skipped unsupported file type: {file}")

    if not documents:
        raise ValueError(
            f"No supported documents (.pdf, .txt) found in '{data_path}'."
        )

    return documents
