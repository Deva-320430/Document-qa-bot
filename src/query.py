"""
Query module.

Loads the persisted vector store and provides a single function,
ask_question(), that:
  1. Retrieves the top-k most relevant chunks for a question
  2. Builds a grounded prompt restricting the LLM to that context
  3. Calls Gemini to generate an answer
  4. Returns the answer along with the source chunks used (for citations)
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from src.rag import load_vector_store

load_dotenv()

DEFAULT_K = 3

_vectordb = None
_llm = None


def _get_vectordb():
    global _vectordb
    if _vectordb is None:
        _vectordb = load_vector_store()
    return _vectordb


def _get_llm():
    global _llm
    if _llm is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GOOGLE_API_KEY not found. Add it to a .env file in the project "
                "root (see .env.example) or export it as an environment variable."
            )
        _llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            google_api_key=api_key,
        )
    return _llm


PROMPT_TEMPLATE = """You are a helpful assistant that answers questions using ONLY the \
provided context from a set of documents.

Rules:
- Answer strictly using the information in the context below.
- If the answer is not present in the context, respond exactly with: \
"I could not find the answer in the documents."
- Do not use outside knowledge, even if you know the answer.
- Be concise and direct.

Context:
{context}

Question:
{question}

Answer:"""


def ask_question(question, k=DEFAULT_K):
    """
    Answer a question using retrieval-augmented generation.

    Args:
        question (str): The user's natural language question.
        k (int): Number of top chunks to retrieve as context.

    Returns:
        tuple[str, list[Document]]: The generated answer text, and the list
            of source Document chunks that were retrieved and used as context.
    """
    vectordb = _get_vectordb()
    llm = _get_llm()

    docs = vectordb.similarity_search(question, k=k)

    if not docs:
        return "I could not find the answer in the documents.", []

    context = "\n\n---\n\n".join(doc.page_content for doc in docs)

    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    response = llm.invoke(prompt)

    return response.content, docs
