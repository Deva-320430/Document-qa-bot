"""
Chunking utilities.

Splits loaded documents into overlapping chunks suitable for embedding
and retrieval. Uses RecursiveCharacterTextSplitter, which tries to split
on paragraph/sentence boundaries first, falling back to smaller separators
only when necessary -- this avoids cutting sentences mid-way wherever possible.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    """
    Split documents into overlapping chunks.

    Args:
        documents (list[Document]): Documents loaded via ingest.load_documents().
        chunk_size (int): Target maximum characters per chunk.
        chunk_overlap (int): Number of overlapping characters between
            consecutive chunks, used to preserve context across chunk
            boundaries.

    Returns:
        list[Document]: Chunked documents. Original metadata (source, page)
            is automatically preserved on each chunk by the splitter.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    # Add a simple incremental chunk_id to metadata for easier debugging/citation.
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    return chunks
