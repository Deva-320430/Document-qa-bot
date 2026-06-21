"""
Vector store module.

Handles embedding model setup and creation/loading of the persistent
ChromaDB vector store.
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
#from langchain_community.vectorstores import Chroma

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
PERSIST_DIRECTORY = "db/chroma_db"
COLLECTION_NAME = "document_qa"


def get_embedding_model():
    """
    Load the sentence-transformers embedding model.

    all-MiniLM-L6-v2 is a small, fast, free, locally-run embedding model
    that produces 384-dimensional embeddings with good general-purpose
    retrieval quality -- a strong default for beginner RAG projects.
    """
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def create_vector_store(chunks, persist_directory=PERSIST_DIRECTORY):
    """
    Build a new persistent Chroma vector store from document chunks.

    Note: As of Chroma 0.4+, persistence is automatic whenever a
    `persist_directory` is supplied -- there is no need to call
    `.persist()` manually (that method is deprecated).

    Args:
        chunks (list[Document]): Chunked documents to embed and index.
        persist_directory (str): Directory where the Chroma DB is stored.

    Returns:
        Chroma: The populated vector store instance.
    """
    embedding_model = get_embedding_model()

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_name=COLLECTION_NAME,
    )

    print(f"Vector database created and persisted to '{persist_directory}'.")
    return vectordb


def load_vector_store(persist_directory=PERSIST_DIRECTORY):
    """
    Load an existing persistent Chroma vector store from disk.

    Returns:
        Chroma: The loaded vector store instance.
    """
    embedding_model = get_embedding_model()

    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
        collection_name=COLLECTION_NAME,
    )

    return vectordb
