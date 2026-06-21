"""
Interactive CLI for the Document Q&A Bot.

Run after main.py has built the vector store:
    python query_bot.py

Type 'exit' or 'quit' to leave.
"""

from src.query import ask_question


def print_sources(docs):
    if not docs:
        return
    print("\nSOURCES:")
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        page_str = f"Page {page + 1}" if isinstance(page, int) else "N/A"
        excerpt = doc.page_content[:200].replace("\n", " ").strip()

        print(f"  [{i}] {source} | {page_str}")
        print(f"      Excerpt: {excerpt}...")


def main():
    print("=" * 60)
    print("  Document Q&A Bot (RAG)")
    print("  Ask questions about the documents in data/")
    print("  Type 'exit' or 'quit' to leave")
    print("=" * 60)

    while True:
        try:
            question = input("\nAsk: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue

        if question.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        try:
            answer, docs = ask_question(question)
        except EnvironmentError as e:
            print(f"\n[Configuration error] {e}")
            continue
        except Exception as e:
            print(f"\n[Error] Something went wrong: {e}")
            continue

        print("\nANSWER:")
        print(answer)

        print_sources(docs)


if __name__ == "__main__":
    main()
